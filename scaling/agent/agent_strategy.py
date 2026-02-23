# =============================================================================
# scaling/agent/agent_strategy.py
# Created:  [Lc-v0.1.1] 2026-02-16
# =============================================================================
"""
Planner-Translator-Summarizer (PTS) scaling strategy.

This strategy decomposes each agent step into three phases:
  1. **Planner**    (best-of-N select)   — produces a step-by-step plan.
  2. **Translator** (best-of-N weighted) — converts the plan into an action.
  3. **Summarizer** (best-of-N select)   — summarizes the step outcome.

It subclasses ScalingStrategy and overrides `select_action` plus the
summarisation step in `step` (to use the Summarizer component).
"""

import time
from typing import Any, Callable, Optional

import numpy as np
from absl import logging

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import m3a_utils
from android_world.agents import agent_utils
from android_world.env import interface
from android_world.env import json_action

from client import LLMClient, Config, OpenAICompatibleVerifierClient
from scaling.base import ScalingStrategy, StepContext, CandidateResult, AggregatedCandidate
from scaling.agent.planner import Planner
from scaling.agent.translator import Translator
from scaling.agent.summarizer import Summarizer


class PlannerTranslatorSummarizerStrategy(ScalingStrategy):
    """Three-phase agent: Planner → Translator → Summarizer.

    Each phase uses a best-of-N strategy:
      - Planner:    N plans generated, verifier selects best m.
      - Translator: N actions generated, verifier scores each, best by
                    total (weighted) score wins.
      - Summarizer: N summaries generated, verifier selects best one.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        actor_llm: infer.MultimodalLlmWrapper,
        # -- Planner settings --
        planner_n: int = 3,
        planner_m: int = 1,
        planner_temperature: float = 0.7,
        # -- Translator settings --
        translator_n: int = 3,
        translator_temperature: float = 0.7,
        # -- Summarizer settings --
        summarizer_n: int = 3,
        summarizer_temperature: float = 0.7,
        # -- Verifier settings (shared) --
        verifier_model: str = "qwen3-vl",
        verifier_config_path: str = "config.yaml",
        verifier_backend: str = "local",
        qwen_api_key: str = "",
        qwen_base_url: str = "",
        **kwargs,
    ):
        """
        Args:
            env:                  AndroidWorld async environment.
            actor_llm:            LLM wrapper for generation (actor).
            planner_n:            Number of candidate plans (N for planner).
            planner_m:            Number of best plans verifier selects (m).
            planner_temperature:  Temperature for plan generation.
            translator_n:         Number of candidate actions (N for translator).
            translator_temperature: Temperature for action generation.
            summarizer_n:         Number of candidate summaries (N for summarizer).
            summarizer_temperature: Temperature for summary generation.
            verifier_model:       Model name for the verifier LLM.
            verifier_config_path: Path to config.yaml for server connection.
            verifier_backend:     "local" or "openai_compatible".
            qwen_api_key:         API key for OpenAI-compatible backend.
            qwen_base_url:        Base URL for OpenAI-compatible backend.
        """
        super().__init__(env, actor_llm, name="PTS_Agent", **kwargs)

        # -- Store verifier config for lazy init --
        self._verifier_backend = verifier_backend
        self._verifier_config_path = verifier_config_path
        self._qwen_api_key = qwen_api_key
        self._qwen_base_url = qwen_base_url
        self._verifier_client_instance: Optional[
            LLMClient | OpenAICompatibleVerifierClient
        ] = None

        # -- Store component settings (init after verifier is ready) --
        self._planner_n = planner_n
        self._planner_m = planner_m
        self._planner_temp = planner_temperature
        self._translator_n = translator_n
        self._translator_temp = translator_temperature
        self._summarizer_n = summarizer_n
        self._summarizer_temp = summarizer_temperature
        self._verifier_model = verifier_model

        # Components are lazy-initialized on first use
        self._planner: Optional[Planner] = None
        self._translator: Optional[Translator] = None
        self._summarizer: Optional[Summarizer] = None

    # -----------------------------------------------------------------
    # Lazy initialization
    # -----------------------------------------------------------------

    @property
    def _verifier(self) -> LLMClient | OpenAICompatibleVerifierClient:
        """Lazy-init the shared verifier client."""
        if self._verifier_client_instance is None:
            if self._verifier_backend == "openai_compatible":
                self._verifier_client_instance = OpenAICompatibleVerifierClient(
                    api_key=self._qwen_api_key or None,
                    base_url=self._qwen_base_url or None,
                )
                logging.info("[PTS] Using OpenAI-compatible verifier backend.")
            else:
                cfg = Config.load(self._verifier_config_path)
                self._verifier_client_instance = LLMClient(cfg)
                logging.info("[PTS] Using local verifier backend.")
        return self._verifier_client_instance

    @property
    def planner(self) -> Planner:
        """Lazy-init the Planner component."""
        if self._planner is None:
            self._planner = Planner(
                llm=self.llm,
                n_samples=self._planner_n,
                m_select=self._planner_m,
                temperature=self._planner_temp,
                verifier_model=self._verifier_model,
                verifier_client=self._verifier,
            )
        return self._planner

    @property
    def translator(self) -> Translator:
        """Lazy-init the Translator component."""
        if self._translator is None:
            self._translator = Translator(
                llm=self.llm,
                n_samples=self._translator_n,
                temperature=self._translator_temp,
                verifier_model=self._verifier_model,
                verifier_client=self._verifier,
            )
        return self._translator

    @property
    def summarizer(self) -> Summarizer:
        """Lazy-init the Summarizer component."""
        if self._summarizer is None:
            self._summarizer = Summarizer(
                llm=self.llm,
                n_samples=self._summarizer_n,
                temperature=self._summarizer_temp,
                verifier_model=self._verifier_model,
                verifier_client=self._verifier,
            )
        return self._summarizer

    # -----------------------------------------------------------------
    # Overrides
    # -----------------------------------------------------------------

    def reset(self, go_home_on_reset: bool = False):
        """Reset agent state for a new task."""
        super().reset(go_home_on_reset)
        self.planner.reset()

    def select_action(self, ctx: StepContext) -> tuple[str, str]:
        """Run the Planner → Translator pipeline.

        Returns:
            (reason, action) where reason is the plan text.
        """
        # --- Phase 1: Plan ---
        plan = self.planner.generate_plan(
            ctx,
            self.additional_guidelines,
            logger=self._planner_logger,
        )
        if not plan:
            logging.warning("[PTS] Planner produced no plan.")
            return ("", "")

        logging.info("[PTS] Plan: %s", plan[:120])

        # --- Phase 2: Translate plan → action ---
        action, candidates, aggregated = self.translator.translate(ctx, plan)
        if not action:
            logging.warning("[PTS] Translator produced no action.")
            return ("", "")

        # Notify step logger (if attached)
        if self._step_logger is not None and candidates:
            self._step_logger(candidates, aggregated)

        return (plan, action)

    # -----------------------------------------------------------------
    # Override step() to use the Summarizer component
    # -----------------------------------------------------------------

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """One agent step: observe → plan → translate → execute → summarize.

        This overrides the base class step() to replace the single-call
        summarisation with the best-of-N Summarizer component.
        """
        step_data: dict[str, Any] = {
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "before_ui_elements": [],
            "after_screenshot_with_som": None,
            "action_prompt": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "action_raw_response": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
        }
        logging.info("----------step %s----------", str(len(self.history) + 1))

        # ---- 1. Observe ----
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_ui_elements = state.ui_elements
        step_data["before_ui_elements"] = before_ui_elements
        before_ui_elements_text = m3a._generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )

        raw_screenshot = state.pixels.copy()
        step_data["raw_screenshot"] = raw_screenshot

        # Build SoM-annotated screenshot
        screenshot_som = state.pixels.copy()
        for idx, elem in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(elem, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    screenshot_som, elem, idx,
                    logical_screen_size, physical_frame_boundary, orientation,
                )
        step_data["before_screenshot_with_som"] = screenshot_som.copy()

        # Formatted history strings
        history_strs = [
            f"Step {i+1}- {h['summary']}" for i, h in enumerate(self.history)
        ]

        ctx = StepContext(
            goal=goal,
            raw_screenshot=raw_screenshot,
            screenshot_with_som=screenshot_som,
            ui_elements=before_ui_elements,
            ui_elements_text=before_ui_elements_text,
            history=history_strs,
            logical_screen_size=logical_screen_size,
            orientation=orientation,
            physical_frame_boundary=physical_frame_boundary,
        )

        # ---- 2. Plan + Translate (via select_action) ----
        reason, action = self.select_action(ctx)

        if (not reason) or (not action):
            logging.info("[PTS] Strategy returned empty reason/action.")
            step_data["summary"] = (
                "Strategy did not return a valid action, so no action performed."
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        logging.info("Action: %s", action)
        logging.info("Plan: %s", reason[:120])
        step_data["action_reason"] = reason
        step_data["action_output"] = f"Plan: {reason}\nAction: {action}"

        # ---- 3. Convert & validate ----
        try:
            converted = json_action.JSONAction(
                **agent_utils.extract_json(action)
            )
            step_data["action_output_json"] = converted
        except Exception as exc:
            logging.info("Failed to convert output to action: %s", exc)
            step_data["summary"] = (
                "Cannot parse output to a valid action. Use correct JSON format!"
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # Index bounds check
        if (
            converted.action_type in ["click", "long_press", "input_text", "scroll"]
            and converted.index is not None
            and converted.index >= len(before_ui_elements)
        ):
            logging.info(
                "Index %s out of range (%d elements).",
                converted.index, len(before_ui_elements),
            )
            step_data["summary"] = "Index out of range for UI element list!"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # Mark targeted element on raw screenshot
        if (
            converted.action_type in ["click", "long_press", "input_text", "scroll"]
            and converted.index is not None
        ):
            m3a_utils.add_ui_element_mark(
                step_data["raw_screenshot"],
                before_ui_elements[converted.index],
                converted.index,
                logical_screen_size, physical_frame_boundary, orientation,
            )

        # ---- 4. Terminal actions ----
        if converted.action_type == "status":
            if converted.goal_status == "infeasible":
                logging.info("Agent thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted.action_type == "answer":
            logging.info("Agent answered: %s", converted.text)

        # ---- 5. Execute ----
        try:
            self.env.execute_action(converted)
        except Exception as exc:
            logging.info("Failed to execute action: %s", exc)
            step_data["summary"] = (
                "Cannot execute action. Ensure correct JSON format!"
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        time.sleep(self.wait_after_action_seconds)

        # ---- 6. Summarize (best-of-N via Summarizer component) ----
        after_state = self.env.get_state(wait_to_stabilize=False)
        after_ui_elements = after_state.ui_elements
        after_ui_text = m3a._generate_ui_elements_description_list(
            after_ui_elements, self.env.logical_screen_size
        )
        after_screenshot = after_state.pixels.copy()
        for idx, elem in enumerate(after_ui_elements):
            if m3a_utils.validate_ui_element(elem, self.env.logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    after_screenshot, elem, idx,
                    self.env.logical_screen_size,
                    self.env.physical_frame_boundary,
                    self.env.orientation,
                )

        m3a_utils.add_screenshot_label(
            step_data["before_screenshot_with_som"], "before"
        )
        m3a_utils.add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        # Use the Summarizer component (best-of-N select)
        summary, summary_prompt = self.summarizer.summarize(
            goal=goal,
            action=action,
            reason=reason,
            before_elements=before_ui_elements_text,
            after_elements=after_ui_text,
            before_screenshot=step_data["before_screenshot_with_som"],
            after_screenshot=after_screenshot,
            logger=self._summarizer_logger,
        )

        if not summary:
            logging.info("[PTS] Summarizer produced no summary.")
            step_data["summary"] = f"Action selected: {action}. (summary failed)"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        step_data["summary_raw_response"] = summary
        logging.info("Summary: %s", summary)

        # Notify summary logger
        if self._summary_logger is not None:
            self._summary_logger(
                len(self.history) + 1, summary_prompt, summary,
            )

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
