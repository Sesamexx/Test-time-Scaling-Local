# =============================================================================
# scaling/baseline.py
# Created:  [Lc-v0.0.3] 2026-02-10
# =============================================================================
"""
Baseline (no scaling) strategy.

Simply delegates to the actor LLM once per step — equivalent to vanilla
M3A but routed through the ScalingStrategy interface.
"""

from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import m3a_utils
from android_world.env import interface

from scaling.base import ScalingStrategy, StepContext, CandidateResult, AggregatedCandidate


class BaselineStrategy(ScalingStrategy):
    """Pass-through strategy: one sample, no verification."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        **kwargs,
    ):
        super().__init__(env, llm, name="Baseline", **kwargs)

    # ---- core hook ----

    def select_action(self, ctx: StepContext) -> tuple[str, str]:
        """Ask the actor LLM once and return its action directly."""
        prompt = m3a._action_selection_prompt(
            ctx.goal,
            ctx.history,
            ctx.ui_elements_text,
            self.additional_guidelines,
        )
        output, is_safe, raw = self.llm.predict_mm(
            prompt, [ctx.raw_screenshot, ctx.screenshot_with_som],
        )

        # Safety filter
        if is_safe is False:
            output = (
                f"Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}\n"
                f'Action: {{"action_type": "status", "goal_status": "infeasible"}}'
            )

        if not raw:
            return ("", "")

        reason, action = m3a_utils.parse_reason_action_output(output)
        reason = reason or ""
        action = action or ""

        # Notify step logger (single candidate, no verifier score)
        if self._step_logger is not None and reason and action:
            cr = CandidateResult(
                reason=reason, action=action,
                score=float('nan'), justification="baseline",
                raw_response=output or "", actor_prompt=prompt,
            )
            agg = AggregatedCandidate(
                action=action, reason=reason,
                total_score=float('nan'), count=1,
                best_justification="baseline", individuals=[cr],
            )
            self._step_logger([cr], [agg])

        return (reason, action)
