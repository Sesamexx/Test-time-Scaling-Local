# =============================================================================
# scaling/agent/translator/translator.py
# Created:  [Lc-v0.1.1] 2026-02-16
# =============================================================================
"""
Translator with best-of-N weighted scoring.

Workflow:
  1. Generator produces N candidate actions given the plan and UI elements.
  2. Verifier scores each action 0-10.
  3. Actions are grouped by action string; total scores are summed.
  4. The action with the highest total score is selected.

The translator outputs ONLY an action JSON (no reason part), because
the reasoning is handled by the planner.
"""

import asyncio
import json
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from absl import logging

from android_world.agents import infer
from client import LLMClient, array_to_base64, OpenAICompatibleVerifierClient
from scaling.base import StepContext, CandidateResult, AggregatedCandidate
from scaling.agent.prompts import (
    TRANSLATOR_PROMPT,
    TRANSLATOR_VERIFIER_PROMPT,
    VALID_ACTIONS_JSON,
)


# ── Programmatic format validation ──────────────────────────────────────
# These checks run BEFORE the LLM verifier to catch obvious format errors
# that the small verifier model often misjudges.

_VALID_ACTION_TYPES = frozenset([
    "open_app", "click", "long_press", "input_text", "scroll",
    "keyboard_enter", "navigate_home", "navigate_back", "wait",
    "answer", "status",
])

# Actions that require an "index" key
_INDEX_REQUIRED = frozenset(["click", "long_press", "input_text"])
# Actions that may optionally have an "index" key
_INDEX_OPTIONAL = frozenset(["scroll"])


def _programmatic_validate(action_str: str, max_index: int) -> tuple[bool, str]:
    """Validate an action string programmatically.

    Returns:
        (is_valid, rejection_reason) — if is_valid is False, the action
        should be scored 0 without consulting the verifier LLM.
    """
    # 1. Must be valid JSON
    try:
        obj = json.loads(action_str)
    except (json.JSONDecodeError, ValueError):
        return (False, "not valid JSON")

    if not isinstance(obj, dict):
        return (False, "JSON is not an object")

    # 2. Must have "action_type"
    atype = obj.get("action_type")
    if not atype:
        return (False, "missing action_type")
    if atype not in _VALID_ACTION_TYPES:
        return (False, f"unknown action_type: {atype}")

    # 3. Must NOT contain "point", "point1", "point2" or coordinate arrays
    for forbidden_key in ("point", "point1", "point2", "element_index"):
        if forbidden_key in obj:
            return (False, f"uses forbidden key '{forbidden_key}'")

    # 4. Check "index" constraints
    if atype in _INDEX_REQUIRED:
        idx = obj.get("index")
        if idx is None:
            return (False, f"'{atype}' requires 'index'")
        if not isinstance(idx, int) or idx < 0 or idx > max_index:
            return (False, f"index {idx} out of range [0, {max_index}]")

    if atype in _INDEX_OPTIONAL:
        idx = obj.get("index")
        if idx is not None and (not isinstance(idx, int) or idx < 0 or idx > max_index):
            return (False, f"index {idx} out of range [0, {max_index}]")

    # 5. Extra keys like "explain" are tolerated but warn-worthy; still valid

    # 6. Special action-specific checks
    if atype == "open_app" and not obj.get("app_name"):
        return (False, "open_app missing app_name")
    if atype == "input_text" and "text" not in obj:
        return (False, "input_text missing text")
    if atype == "scroll" and "direction" not in obj:
        return (False, "scroll missing direction")
    if atype == "answer" and "text" not in obj:
        return (False, "answer missing text")
    if atype == "status" and "goal_status" not in obj:
        return (False, "status missing goal_status")

    return (True, "")


def _extract_max_index(ui_elements_text: str) -> int:
    """Extract the maximum UI element index from the UI elements text."""
    max_idx = -1
    for m in re.finditer(r'UI element (\d+):', ui_elements_text):
        idx = int(m.group(1))
        if idx > max_idx:
            max_idx = idx
    return max_idx if max_idx >= 0 else 9999  # fallback: don't filter


def _extract_first_step(plan: str) -> str:
    """Parse a numbered plan and return only the first step text.

    Handles formats like "1. Open Settings" or "1) Open Settings".
    Falls back to the first non-empty line if no numbering is found.
    """
    for line in plan.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "1. ...", "1) ...", "1: ..." etc.
        m = re.match(r'^\d+[\.\)\:\-]\s*(.*)', line)
        if m:
            return m.group(1).strip()
        # If no numbering found, return the first non-empty line
        return line
    return plan.strip()


class Translator:
    """Translate a plan into a concrete action via best-of-N weighted.

    Attributes:
        llm:              Actor LLM wrapper (generates candidate actions).
        n_samples:        Number of candidate actions to generate (N).
        temperature:      Sampling temperature for action generation.
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
    # Action generation (single sample)
    # -----------------------------------------------------------------

    def _build_prompt(self, ctx: StepContext, plan: str) -> str:
        """Build the translator prompt from context and plan.

        Only the first step of the plan is passed to the translator,
        since it only needs to produce the immediate next action.
        """
        first_step = _extract_first_step(plan)
        return TRANSLATOR_PROMPT.format(
            plan=first_step,
            ui_elements=ctx.ui_elements_text or "Not available",
            valid_actions=VALID_ACTIONS_JSON,
        )

    def _sample_one(
        self,
        prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, str]:
        """Generate one candidate action from the actor LLM.

        Returns:
            (action_string, raw_output)
        """
        old_temp = getattr(self.llm, "temperature", None)
        if old_temp is not None:
            self.llm.temperature = self.temperature

        output, is_safe, raw = self.llm.predict_mm(prompt, images)

        if old_temp is not None:
            self.llm.temperature = old_temp

        if not raw or is_safe is False:
            return ("", "")

        # Extract JSON action from the output
        text = (output or "").strip()
        # Try to find a JSON object in the output
        json_match = re.search(r'\{[^{}]*\}', text)
        if not json_match:
            return (text, text)

        candidate = json_match.group(0)

        # Validate it's actually parseable JSON; if not, return raw text
        # (the programmatic validator will reject it later)
        try:
            obj = json.loads(candidate)
            # Re-serialize to normalise (removes trailing commas, etc.)
            # Also strip extra keys that are not part of the action schema
            # to help with aggregation (grouping identical actions)
            known_keys = {
                "action_type", "index", "app_name", "text",
                "direction", "goal_status",
            }
            cleaned = {k: v for k, v in obj.items() if k in known_keys}
            candidate = json.dumps(cleaned, separators=(",", ":"))
        except (json.JSONDecodeError, ValueError):
            # Leave it as-is; the programmatic validator will catch it
            pass

        return (candidate, text)

    # -----------------------------------------------------------------
    # Verifier: score one candidate action
    # -----------------------------------------------------------------

    async def _score_candidate(
        self,
        ctx: StepContext,
        first_step: str,
        action: str,
    ) -> tuple[float, str]:
        """Ask verifier to score a candidate action.

        Runs programmatic format validation first. If the action fails
        basic format checks, it gets score 0 immediately without calling
        the LLM verifier. This catches issues that small verifier models
        frequently misjudge (e.g., "point" coordinates, out-of-range index).

        Returns:
            (score, raw_verifier_response) — score in 0-10.
        """
        # ── Programmatic pre-filter ──
        max_idx = _extract_max_index(ctx.ui_elements_text or "")
        is_valid, reject_reason = _programmatic_validate(action, max_idx)
        if not is_valid:
            logging.info(
                "[Translator] Programmatic reject: %s | action=%s",
                reject_reason, action[:80],
            )
            return (0.0, f"programmatic reject: {reject_reason}")

        # ── Special-case: step requires a system action ──
        step_lower = first_step.lower()
        try:
            obj = json.loads(action)
        except Exception:
            return (0.0, "programmatic reject: not valid JSON")

        if "mark task as complete" in step_lower:
            if obj.get("action_type") != "status" or obj.get("goal_status") != "complete":
                return (0.0, "programmatic reject: step requires status/complete")
        elif "mark task as infeasible" in step_lower:
            if obj.get("action_type") != "status" or obj.get("goal_status") != "infeasible":
                return (0.0, "programmatic reject: step requires status/infeasible")
        elif "provide an answer" in step_lower:
            if obj.get("action_type") != "answer":
                return (0.0, "programmatic reject: step requires answer action")

        # ── LLM verifier (semantic quality) ──
        if self.verifier_client is None:
            return (5.0, "no verifier")

        prompt = TRANSLATOR_VERIFIER_PROMPT.format(
            plan=first_step,
            action=action,
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

        try:
            resp = await self.verifier_client.chat(
                model=self.verifier_model,
                messages=messages,
                temperature=0.0,
                max_tokens=16,
            )
            match = re.search(r'\b(\d{1,2})\b', resp.strip())
            if match:
                score = min(float(match.group(1)), 10.0)
                return (score, resp)
            logging.warning("[Translator] Verifier returned no score: %s", resp[:100])
            return (5.0, resp)
        except Exception as exc:
            logging.warning("[Translator] Verifier call failed: %s", exc)
            return (5.0, f"error: {exc}")

    # -----------------------------------------------------------------
    # Parallel sample + score
    # -----------------------------------------------------------------

    async def _sample_and_score_one(
        self,
        idx: int,
        prompt: str,
        images: list[np.ndarray],
        ctx: StepContext,
        first_step: str,
    ) -> tuple[int, str, float, str, str]:
        """Sample one action and score it. Runs as an async coroutine.

        Returns:
            (index, action, score, raw_output, verifier_response)
        """
        loop = asyncio.get_event_loop()
        action, raw_output = await loop.run_in_executor(
            None, self._sample_one, prompt, images,
        )
        if not action:
            return (idx, "", 0.0, "", "")

        score, verifier_resp = await self._score_candidate(ctx, first_step, action)
        return (idx, action, score, raw_output, verifier_resp)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def translate(
        self,
        ctx: StepContext,
        plan: str,
    ) -> tuple[str, list[CandidateResult], list[AggregatedCandidate]]:
        """Generate N actions, score each, return best by total score.

        Args:
            ctx:  Step context (screenshots, UI elements, etc.).
            plan: The plan from the planner.

        Returns:
            (best_action, candidate_results, aggregated_candidates)
        """
        first_step = _extract_first_step(plan)

        # ── LLM-based best-of-N ──
        prompt = self._build_prompt(ctx, plan)
        images = [ctx.raw_screenshot, ctx.screenshot_with_som]

        # Get or create event loop
        loop = self._get_event_loop()

        # --- Launch N sample+score pipelines in parallel ---
        coros = [
            self._sample_and_score_one(i, prompt, images, ctx, first_step)
            for i in range(self.n_samples)
        ]
        results = loop.run_until_complete(asyncio.gather(*coros))

        # --- Collect valid candidates ---
        candidate_results: list[CandidateResult] = []
        for (idx, action, score, raw_output, verifier_resp) in results:
            logging.info(
                "[Translator] Candidate %d/%d: score=%.1f action=%s",
                idx + 1, self.n_samples, score, action[:80] if action else "",
            )
            if action:
                candidate_results.append(CandidateResult(
                    reason=plan,       # Use plan as the "reason"
                    action=action,
                    score=score,
                    justification="",
                    raw_response=raw_output,
                    actor_prompt=prompt,
                    verifier_response=verifier_resp,
                ))

        if not candidate_results:
            logging.warning("[Translator] No valid candidates.")
            return ("", [], [])

        # Sort by score descending
        candidate_results.sort(key=lambda c: c.score, reverse=True)

        # --- Aggregate: group by action string, sum scores ---
        groups: dict[str, list[CandidateResult]] = defaultdict(list)
        for cr in candidate_results:
            groups[cr.action].append(cr)

        aggregated: list[AggregatedCandidate] = []
        for action_str, members in groups.items():
            best_member = max(members, key=lambda c: c.score)
            aggregated.append(AggregatedCandidate(
                action=action_str,
                reason=plan,
                total_score=sum(m.score for m in members),
                count=len(members),
                best_justification="",
                individuals=members,
            ))
        aggregated.sort(key=lambda a: a.total_score, reverse=True)

        best = aggregated[0]
        logging.info(
            "[Translator] Selected best (total=%.1f, count=%d): %s",
            best.total_score, best.count, best.action,
        )
        return (best.action, candidate_results, aggregated)

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
