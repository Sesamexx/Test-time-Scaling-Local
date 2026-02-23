# =============================================================================
# scaling/agent/summarizer/summarizer.py
# Created:  [Lc-v0.1.1] 2026-02-16
# =============================================================================
"""
Summarizer with best-of-N selection.

Workflow:
  1. Generator produces N candidate summaries of the latest step.
  2. Verifier selects the single best summary.

This mirrors the existing summarization logic but applies best-of-N
to improve summary quality.
"""

import asyncio
import re
from typing import Callable, Optional

import numpy as np
from absl import logging

from android_world.agents import infer
from client import LLMClient, array_to_base64, OpenAICompatibleVerifierClient
from scaling.agent.prompts import (
    SUMMARIZER_PROMPT,
    SUMMARIZER_VERIFIER_PROMPT,
)


class Summarizer:
    """Summarize a step via best-of-N selection.

    Attributes:
        llm:              LLM wrapper (generates candidate summaries).
        n_samples:        Number of candidate summaries to generate (N).
        temperature:      Sampling temperature for summary generation.
        verifier_model:   Model name for the verifier.
        verifier_client:  Client for the verifier LLM.
    """

    def __init__(
        self,
        llm: infer.MultimodalLlmWrapper,
        n_samples: int = 3,
        temperature: float = 0.7,
        verifier_model: str = "qwen3-vl",
        verifier_client: Optional[LLMClient | OpenAICompatibleVerifierClient] = None,
    ):
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature
        self.verifier_model = verifier_model
        self.verifier_client = verifier_client

    # -----------------------------------------------------------------
    # Summary generation (single sample)
    # -----------------------------------------------------------------

    def _build_prompt(
        self,
        goal: str,
        action: str,
        reason: str,
        before_elements: str,
        after_elements: str,
    ) -> str:
        """Build the summarizer prompt."""
        return SUMMARIZER_PROMPT.format(
            goal=goal,
            action=action,
            reason=reason,
            before_elements=before_elements,
            after_elements=after_elements,
        )

    def _sample_one(
        self,
        prompt: str,
        images: list[np.ndarray],
    ) -> str:
        """Generate one candidate summary."""
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
    # Verifier: select best summary
    # -----------------------------------------------------------------

    async def _verify_and_select(
        self,
        goal: str,
        action: str,
        candidates: list[str],
        images: list[np.ndarray],
    ) -> tuple[int, str]:
        """Ask verifier to pick the best summary.

        Returns:
            (0-based index of selected summary, raw_verifier_response)
        """
        if self.verifier_client is None:
            return (0, "(no verifier)")

        # Format candidates
        candidates_text = ""
        for i, summary in enumerate(candidates):
            candidates_text += f"\n--- Summary {i + 1} ---\n{summary}\n"

        prompt = SUMMARIZER_VERIFIER_PROMPT.format(
            goal=goal,
            action=action,
            candidates=candidates_text,
        )

        # Build multimodal message
        images_b64 = [array_to_base64(img) for img in images]
        image_contents = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in images_b64
        ]
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *image_contents],
        }]

        try:
            resp = await self.verifier_client.chat(
                model=self.verifier_model,
                messages=messages,
                temperature=0.0,
                max_tokens=16,
            )
            match = re.search(r'\d+', resp.strip())
            if match:
                idx = int(match.group(0)) - 1  # Convert to 0-based
                if 0 <= idx < len(candidates):
                    return (idx, resp.strip())
            logging.warning("[Summarizer] Verifier returned invalid index: %s", resp[:100])
            return (0, resp.strip())
        except Exception as exc:
            logging.warning("[Summarizer] Verifier call failed: %s", exc)
            return (0, f"error: {exc}")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def summarize(
        self,
        goal: str,
        action: str,
        reason: str,
        before_elements: str,
        after_elements: str,
        before_screenshot: np.ndarray,
        after_screenshot: np.ndarray,
        logger: Callable | None = None,
    ) -> tuple[str, str]:
        """Generate N summaries, verify, return the best.

        Args:
            goal:              The task goal.
            action:            The action that was performed.
            reason:            The reason/plan behind the action.
            before_elements:   UI elements text before the action.
            after_elements:    UI elements text after the action.
            before_screenshot: Screenshot before action (with "before" label).
            after_screenshot:  Screenshot after action (with "after" label).
            logger:            Optional callback (candidates, selected_index,
                               verifier_response) -> None for detailed logging.

        Returns:
            (best_summary, summary_prompt) — the selected summary and the
            prompt used for generation.
        """
        prompt = self._build_prompt(goal, action, reason, before_elements, after_elements)
        images = [before_screenshot, after_screenshot]

        # --- Generate N candidate summaries ---
        candidates: list[str] = []
        for i in range(self.n_samples):
            summary = self._sample_one(prompt, images)
            if summary:
                candidates.append(summary)
                logging.info(
                    "[Summarizer] Candidate %d/%d: %s",
                    i + 1, self.n_samples, summary[:80],
                )

        if not candidates:
            logging.warning("[Summarizer] No valid summaries generated.")
            return ("", prompt)

        # If only one candidate or no verifier, skip verification
        if len(candidates) == 1 or self.verifier_client is None:
            verifier_resp = "(skipped — single candidate or no verifier)"
            selected_idx = 0
        else:
            # --- Verify and select best ---
            loop = self._get_event_loop()
            selected_idx, verifier_resp = loop.run_until_complete(
                self._verify_and_select(goal, action, candidates, images)
            )

        logging.info(
            "[Summarizer] Verifier selected summary %d", selected_idx + 1
        )

        # Notify logger if attached
        if logger is not None:
            logger(candidates, selected_idx, verifier_resp)

        return (candidates[selected_idx], prompt)

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
