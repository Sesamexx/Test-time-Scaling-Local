# =============================================================================
# scaling/agent/planner/planner.py
# Created:  [Lc-v0.1.1] 2026-02-15
# =============================================================================
"""
Planner with best-of-N selection.

Workflow:
  1. Generator produces N candidate plans (with temperature > 0).
  2. Verifier selects the best m plans from the N candidates.
  3. The top-ranked plan is used for the current step.

The planner is informed of the previous plan so it can revise rather
than start from scratch, and is told not to retry the same failing
action more than 3 times.
"""

import asyncio
import re
from typing import Callable, Optional

import numpy as np
from absl import logging

from android_world.agents import infer
from client import LLMClient, Config, array_to_base64, OpenAICompatibleVerifierClient
from scaling.base import StepContext
from scaling.agent.prompts import (
    PLANNER_PROMPT,
    PLANNER_VERIFIER_PROMPT,
    VALID_ACTIONS_JSON,
)


class Planner:
    """Generate and select step-by-step plans via best-of-N selection.

    Attributes:
        llm:              Actor LLM wrapper (used to generate plans).
        n_samples:        Number of candidate plans to generate (N).
        m_select:         Number of best plans the verifier selects (m).
        temperature:      Sampling temperature for plan generation.
        previous_plan:    The plan from the previous step (for revision).
    """

    def __init__(
        self,
        llm: infer.MultimodalLlmWrapper,
        n_samples: int = 3,
        m_select: int = 1,
        temperature: float = 0.7,
        verifier_model: str = "qwen3-vl",
        verifier_client: Optional[LLMClient | OpenAICompatibleVerifierClient] = None,
    ):
        self.llm = llm
        self.n_samples = n_samples
        self.m_select = m_select
        self.temperature = temperature
        self.verifier_model = verifier_model
        self.verifier_client = verifier_client
        self.previous_plan: str = "(no previous plan)"

    # -----------------------------------------------------------------
    # Plan generation (single sample)
    # -----------------------------------------------------------------

    def _build_prompt(self, ctx: StepContext, guidelines: list[str] | None) -> str:
        """Build the planner prompt from context."""
        history_str = "\n".join(ctx.history) if ctx.history else "(none yet)"
        guidelines_str = ""
        if guidelines:
            guidelines_str = "\n".join(f"- {g}" for g in guidelines)
        else:
            guidelines_str = "(none)"

        return PLANNER_PROMPT.format(
            goal=ctx.goal,
            valid_actions=VALID_ACTIONS_JSON,
            history=history_str,
            previous_plan=self.previous_plan,
            guidelines=guidelines_str,
        )

    def _sample_one(
        self,
        prompt: str,
        images: list[np.ndarray],
    ) -> str:
        """Generate one candidate plan from the actor LLM."""
        old_temp = getattr(self.llm, "temperature", None)
        if old_temp is not None:
            self.llm.temperature = self.temperature

        output, is_safe, raw = self.llm.predict_mm(prompt, images)

        if old_temp is not None:
            self.llm.temperature = old_temp

        if not raw or is_safe is False:
            return ""
        return (output or "").strip()

    # -----------------------------------------------------------------
    # Verifier: select best m plans
    # -----------------------------------------------------------------

    async def _verify_and_select(
        self,
        ctx: StepContext,
        candidates: list[str],
    ) -> tuple[list[int], str]:
        """Ask verifier to pick the best m plans from candidates.

        Returns:
            (list_of_0based_indices, raw_verifier_response)
        """
        if self.verifier_client is None:
            # No verifier → return first candidate
            return ([0], "(no verifier)")

        # Format candidates for the verifier
        candidates_text = ""
        for i, plan in enumerate(candidates):
            candidates_text += f"\n--- Plan {i + 1} ---\n{plan}\n"

        history_str = "\n".join(ctx.history) if ctx.history else "(none yet)"

        prompt = PLANNER_VERIFIER_PROMPT.format(
            goal=ctx.goal,
            history=history_str,
            candidates=candidates_text,
            valid_actions=VALID_ACTIONS_JSON,
            m=self.m_select,
        )

        # Build multimodal message
        images_b64 = [
            array_to_base64(ctx.raw_screenshot),
            array_to_base64(ctx.screenshot_with_som),
        ]
        image_contents = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in images_b64
        ]
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *image_contents],
        }]

        resp = ""
        try:
            resp = await self.verifier_client.chat(
                model=self.verifier_model,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
            )
            # Parse comma-separated integers from response
            numbers = re.findall(r'\d+', resp.strip())
            # Convert to 0-based indices
            indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(candidates)]
            if indices:
                return (indices[:self.m_select], resp.strip())
        except Exception as exc:
            logging.warning("[Planner] Verifier call failed: %s", exc)
            return ([0], f"error: {exc}")

        # Fallback: return first candidate (verifier returned unparseable response)
        return ([0], resp.strip() if resp else "(no response)")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_plan(
        self,
        ctx: StepContext,
        guidelines: list[str] | None = None,
        logger: Callable | None = None,
    ) -> str:
        """Generate N plans, verify, and return the best one.

        Also updates `self.previous_plan` for the next step.

        Args:
            ctx:        Step context.
            guidelines: Additional task guidelines.
            logger:     Optional callback (candidates, selected_indices,
                        verifier_response) -> None for detailed logging.

        Returns:
            The selected plan as a string.
        """
        prompt = self._build_prompt(ctx, guidelines)
        images = [ctx.raw_screenshot, ctx.screenshot_with_som]

        # --- Generate N candidate plans ---
        candidates: list[str] = []
        for i in range(self.n_samples):
            plan = self._sample_one(prompt, images)
            if plan:
                candidates.append(plan)
                logging.info(
                    "[Planner] Candidate %d/%d: %s",
                    i + 1, self.n_samples, plan[:80],
                )

        if not candidates:
            logging.warning("[Planner] No valid plans generated.")
            return ""

        # If only one candidate or no verifier, skip verification
        if len(candidates) == 1 or self.verifier_client is None:
            selected = candidates[0]
            verifier_resp = "(skipped — single candidate or no verifier)"
            selected_indices = [0]
        else:
            # --- Verify and select best m ---
            loop = self._get_event_loop()
            selected_indices, verifier_resp = loop.run_until_complete(
                self._verify_and_select(ctx, candidates)
            )
            selected = candidates[selected_indices[0]]
            logging.info(
                "[Planner] Verifier selected plan index %d",
                selected_indices[0] + 1,
            )

        # Notify logger if attached
        if logger is not None:
            logger(candidates, selected_indices, verifier_resp)

        # Update previous plan for next step
        self.previous_plan = selected
        return selected

    def reset(self):
        """Reset planner state for a new task."""
        self.previous_plan = "(no previous plan)"

    # -----------------------------------------------------------------
    # Event loop helper
    # -----------------------------------------------------------------

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
