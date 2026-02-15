# =============================================================================
# scaling/best_of_n_weighted.py
# Created:  [Lc-v0.0.3] 2026-02-10
# =============================================================================
"""
Best-of-N Weighted test-time scaling strategy.

Workflow (per step):
  1. Sample N candidate actions from the actor model (gelab-zero-4b-preview)
     using temperature > 0.
  2. Send each candidate — together with the screenshot and goal — to a
     verifier model (qwen3-vl) that returns a 0-10 quality score.
     The verifier also penalises bad output format (unparseable JSON → 0).
  3. Pick the candidate with the highest verifier score.

Actor sampling and verification are performed in parallel (asyncio).
"""

import asyncio
import re
from typing import Optional

import numpy as np
from absl import logging

from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import m3a_utils
from android_world.env import interface

from collections import defaultdict

from client import LLMClient, Config, array_to_base64, OpenAICompatibleVerifierClient
from scaling.base import ScalingStrategy, StepContext, CandidateResult, AggregatedCandidate


# =============================================================================
# Verifier prompt — includes output-format awareness
# =============================================================================

_VALID_ACTION_TYPES = (
    "click", "long_press", "input_text", "keyboard_enter",
    "navigate_home", "navigate_back", "scroll", "open_app",
    "status", "wait", "answer", "double_tap", "swipe",
)

_VERIFIER_PROMPT = """\
You are a quality-assurance verifier for an Android phone agent.

## Context
- **Goal:** {goal}
- **Action history so far:** {history}
- **Available UI elements:** {ui_elements}

## Candidate action to evaluate
- **Reason:** {reason}
- **Action:** {action}

## Instructions
You are given two screenshots: the original screen and the same screen \
annotated with bounding boxes and numeric indexes on UI elements.

Evaluate the candidate on TWO dimensions:

### A. Output format correctness
The action MUST be a valid JSON object with at minimum an "action_type" key. \

All valid action types (score 0-10):
Valid 1. `{{"action_type": "status", "goal_status": "complete"}}`
Valid 2. `{{"action_type": "status", "goal_status": "infeasible"}}`
Valid 3. `{{"action_type": "answer", "text": "<answer_text>"}}`
Valid 4. `{{"action_type": "click", "index": <target_index>}}`
Valid 5. `{{"action_type": "long_press", "index": <target_index>}}`
Valid 6. `{{"action_type": "input_text", "text": <text_input>, "index": <target_index>}}`
Valid 7. `{{"action_type": "keyboard_enter"}}`
Valid 8. `{{"action_type": "navigate_home"}}`
Valid 9. `{{"action_type": "navigate_back"}}`
Valid 10. `{{"action_type": "scroll", "direction": <up, down, left, right>, "index": <optional_target_index>}}`
Valid 11. `{{"action_type": "open_app", "app_name": <name>}}`
Valid 12. `{{"action_type": "wait"}}`

Examples of invalid actions (score 0-3):
Invalid 1. `{{"action_type": "click", "point": [<x>, <y>]}}` Reason: should use "index" instead of "point".
Invalid 2. `{{"action_type": "click", "element_index": <target_index>}}` Reason: should use "index" instead of "element_index".
Invalid 3. `{{"action_type": "input_text", "text": <text_input>}}` Reason: missing "index".
Invalid 4. `{{"action_type": "long_press", "element_index": <target_index>}}` Reason: should use "index" instead of "element_index".

Scoring rules for format:
- If the action string cannot be parsed as JSON at all → score = 0.
- If "action_type" is missing or not one of the valid types → score ≤ 2.
- If required keys are missing (e.g. "click" without "index", "input_text" \
  without "text" and "index", "scroll" without "direction", "open_app" \
  without "app_name", "status" without "goal_status", "answer" without \
  "text") → score ≤ 3.
- If the format is fully correct, proceed to evaluate semantic quality.

### B. Semantic quality (only if format is correct)
1. Is the action type appropriate for the current screen state?
2. Does the target element (if any) match the stated reason?
3. Is this action a logical next step given the history?
4. Could the action cause errors or move away from the goal?

Combine both dimensions into a single score from 0-10 where:
  0 = unparseable / garbage output
  1-3 = format issues or clearly wrong action
  4-6 = plausible but sub-optimal
  7-9 = good, logical, well-formatted
  10 = clearly correct, efficient, and well-formatted

## Output format
Reply with ONLY a single integer (0-10). No other text, no explanation.
"""


# =============================================================================
# Strategy
# =============================================================================

class BestOfNWeightedStrategy(ScalingStrategy):
    """Sample N actions from the actor, score with verifier, pick the best.

    Actor calls and their corresponding verifier calls run in parallel.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        actor_llm: infer.MultimodalLlmWrapper,
        n_samples: int = 3,
        actor_temperature: float = 0.7,
        verifier_model: str = "qwen3-vl",
        verifier_config_path: str = "config.yaml",
        verifier_backend: str = "local",
        qwen_api_key: str = "",
        qwen_base_url: str = "",
        **kwargs,
    ):
        """
        Args:
            env:                 AndroidWorld async environment.
            actor_llm:           LLM wrapper used to *sample* candidate actions.
            n_samples:           Number of candidate actions to sample (N).
            actor_temperature:   Temperature override for the actor during
                                 sampling.
            verifier_model:      Model name for the verifier (default qwen3-vl).
            verifier_config_path: Path to config.yaml for server connection.
            verifier_backend:    "local" to use the LLMClient (local/SSH server)
                                 or "openai_compatible" to use the OpenAI SDK
                                 with QWEN_API_KEY / QWEN_BASE_URL from .env.
            qwen_api_key:        Explicit API key (overrides .env).
            qwen_base_url:       Explicit base URL (overrides .env).
        """
        super().__init__(env, actor_llm, name="BestOfN_Weighted", **kwargs)
        self.n_samples = n_samples
        self.actor_temperature = actor_temperature
        self.verifier_model = verifier_model
        self._verifier_backend = verifier_backend
        self._verifier_client: Optional[LLMClient] = None
        self._openai_verifier_client: Optional[OpenAICompatibleVerifierClient] = None
        self._verifier_config_path = verifier_config_path
        self._qwen_api_key = qwen_api_key
        self._qwen_base_url = qwen_base_url

    @property
    def verifier_client(self) -> LLMClient | OpenAICompatibleVerifierClient:
        """Lazy-init verifier client based on configured backend."""
        if self._verifier_backend == "openai_compatible":
            if self._openai_verifier_client is None:
                self._openai_verifier_client = OpenAICompatibleVerifierClient(
                    api_key=self._qwen_api_key or None,
                    base_url=self._qwen_base_url or None,
                )
                logging.info(
                    "[BestOfN] Using OpenAI-compatible verifier backend."
                )
            return self._openai_verifier_client
        else:
            # Original local backend
            if self._verifier_client is None:
                cfg = Config.load(self._verifier_config_path)
                self._verifier_client = LLMClient(cfg)
                logging.info("[BestOfN] Using local verifier backend.")
            return self._verifier_client

    # ---- internal helpers ----

    def _build_action_prompt(self, ctx: StepContext) -> str:
        """Build the M3A action-selection prompt from context."""
        return m3a._action_selection_prompt(
            ctx.goal,
            ctx.history,
            ctx.ui_elements_text,
            self.additional_guidelines,
        )

    def _sample_one(
        self,
        prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, str, str]:
        """Sample a single (reason, action, raw_output) from the actor LLM."""
        old_temp = getattr(self.llm, "temperature", None)
        if old_temp is not None:
            self.llm.temperature = self.actor_temperature

        output, is_safe, raw = self.llm.predict_mm(prompt, images)

        if old_temp is not None:
            self.llm.temperature = old_temp

        if is_safe is False:
            output = (
                f"Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}\n"
                f'Action: {{"action_type": "status", "goal_status": "infeasible"}}'
            )
        if not raw:
            return ("", "", "")

        reason, action = m3a_utils.parse_reason_action_output(output)
        return (reason or "", action or "", output or "")

    async def _score_candidate_async(
        self,
        ctx: StepContext,
        reason: str,
        action: str,
    ) -> tuple[float, str, str]:
        """Ask the verifier to score one candidate action.

        Returns:
            (score, justification, raw_verifier_response)  — score in 0-10.
        """
        history_str = "\n".join(ctx.history) if ctx.history else "(none yet)"
        prompt = _VERIFIER_PROMPT.format(
            goal=ctx.goal,
            history=history_str,
            ui_elements=ctx.ui_elements_text or "Not available",
            reason=reason,
            action=action,
            valid_action_types=", ".join(_VALID_ACTION_TYPES),
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
            # The verifier should reply with just a single integer 0-10.
            # Extract the first integer found in the response.
            match = re.search(r'\b(\d{1,2})\b', resp.strip())
            if match:
                score = min(float(match.group(1)), 10.0)
                return score, "", resp
            logging.warning("Verifier response has no score: %s", resp[:200])
            return 5.0, "no score parsed", resp
        except Exception as exc:
            logging.warning("Verifier call failed: %s", exc)
            return 5.0, f"error: {exc}", f"error: {exc}"

    async def _sample_and_score_one(
        self,
        idx: int,
        prompt: str,
        images: list[np.ndarray],
        ctx: StepContext,
    ) -> tuple[int, str, str, float, str, str, str, str]:
        """Sample one candidate from actor, then score it with verifier.

        Runs in a single coroutine so N of these can execute concurrently.

        Returns:
            (index, reason, action, score, justification,
             raw_output, actor_prompt, verifier_response)
        """
        # Actor call (sync LLM wrapper → run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        reason, action, raw_output = await loop.run_in_executor(
            None, self._sample_one, prompt, images,
        )
        if not reason or not action:
            return (idx, "", "", 0.0, "empty candidate", "", prompt, "")

        # Verifier call (async)
        score, justification, verifier_resp = await self._score_candidate_async(
            ctx, reason, action,
        )
        return (idx, reason, action, score, justification,
                raw_output, prompt, verifier_resp)

    # ---- core hook ----

    def select_action(self, ctx: StepContext) -> tuple[str, str]:
        """Sample N candidates in parallel, score each, return the best."""
        prompt = self._build_action_prompt(ctx)
        images = [ctx.raw_screenshot, ctx.screenshot_with_som]

        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Launch all N sample+score pipelines concurrently
        coros = [
            self._sample_and_score_one(i, prompt, images, ctx)
            for i in range(self.n_samples)
        ]
        results = loop.run_until_complete(asyncio.gather(*coros))

        # Collect valid candidates
        candidate_results: list[CandidateResult] = []
        for (idx, reason, action, score, justification,
             raw_output, actor_prompt, verifier_resp) in results:
            logging.info(
                "[BestOfN] Candidate %d/%d: score=%.1f action=%s",
                idx + 1, self.n_samples, score, action[:80] if action else "",
            )
            if reason and action:
                candidate_results.append(CandidateResult(
                    reason=reason,
                    action=action,
                    score=score,
                    justification=justification,
                    raw_response=raw_output,
                    actor_prompt=actor_prompt,
                    verifier_response=verifier_resp,
                ))

        if not candidate_results:
            logging.warning("[BestOfN] No valid candidates sampled.")
            return ("", "")

        # Sort individual candidates by score descending
        candidate_results.sort(key=lambda c: c.score, reverse=True)

        # Aggregate: group by action string, sum scores
        groups: dict[str, list[CandidateResult]] = defaultdict(list)
        for cr in candidate_results:
            groups[cr.action].append(cr)

        aggregated: list[AggregatedCandidate] = []
        for action_str, members in groups.items():
            best_member = max(members, key=lambda c: c.score)
            aggregated.append(AggregatedCandidate(
                action=action_str,
                reason=best_member.reason,
                total_score=sum(m.score for m in members),
                count=len(members),
                best_justification=best_member.justification,
                individuals=members,
            ))
        aggregated.sort(key=lambda a: a.total_score, reverse=True)

        # Notify the step logger (if attached)
        if self._step_logger is not None:
            self._step_logger(candidate_results, aggregated)

        # Pick the action with the highest *aggregated* score
        best = aggregated[0]
        logging.info(
            "[BestOfN] Selected best (total_score %.1f, count %d): %s",
            best.total_score, best.count, best.action,
        )
        return (best.reason, best.action)
