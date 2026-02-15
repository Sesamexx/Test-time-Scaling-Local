# =============================================================================
# scaling/base.py
# Created:  [Lc-v0.0.3] 2026-02-10
# =============================================================================
"""
Base class for test-time scaling strategies.

A ScalingStrategy wraps an M3A agent and intercepts each step to apply
a scaling method (e.g., best-of-N, beam search) before committing an
action to the environment. It subclasses EnvironmentInteractingAgent so
it is a drop-in replacement in the existing benchmark runner.
"""

import abc
import time
from dataclasses import dataclass
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
from android_world.env import representation_utils


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class CandidateResult:
    """One scored candidate produced by a scaling strategy.

    Used by the logger to show per-candidate detail.
    """
    reason: str
    action: str
    score: float
    justification: str = ""
    raw_response: str = ""       # full LLM response text (actor)
    actor_prompt: str = ""       # prompt sent to the actor
    verifier_response: str = ""  # full verifier response text


@dataclass
class AggregatedCandidate:
    """A unique action with its total (summed) score and constituents."""
    action: str
    reason: str            # reason from the highest-scoring instance
    total_score: float
    count: int
    best_justification: str = ""
    individuals: list = None  # list[CandidateResult]

    def __post_init__(self):
        if self.individuals is None:
            self.individuals = []

class StepContext:
    """All information available at one decision point.

    This is the *only* object a strategy needs to decide on an action.
    It deliberately excludes the prompt — strategies build their own.

    Attributes:
        goal:                   Natural-language task goal.
        raw_screenshot:         Original screenshot (H×W×3 uint8 ndarray).
        screenshot_with_som:    Screenshot annotated with SoM bounding boxes.
        ui_elements:            Parsed UI element list.
        ui_elements_text:       Pre-rendered textual description of ui_elements.
        history:                List of summary strings for past steps.
        logical_screen_size:    (width, height) in logical pixels.
        orientation:            Screen orientation (0-3).
        physical_frame_boundary: (x0, y0, x1, y1) of the physical frame.
    """

    __slots__ = (
        "goal",
        "raw_screenshot",
        "screenshot_with_som",
        "ui_elements",
        "ui_elements_text",
        "history",
        "logical_screen_size",
        "orientation",
        "physical_frame_boundary",
    )

    def __init__(
        self,
        goal: str,
        raw_screenshot: np.ndarray,
        screenshot_with_som: np.ndarray,
        ui_elements: list[representation_utils.UIElement],
        ui_elements_text: str,
        history: list[str],
        logical_screen_size: tuple[int, int],
        orientation: int,
        physical_frame_boundary: tuple[int, int, int, int],
    ):
        self.goal = goal
        self.raw_screenshot = raw_screenshot
        self.screenshot_with_som = screenshot_with_som
        self.ui_elements = ui_elements
        self.ui_elements_text = ui_elements_text
        self.history = history
        self.logical_screen_size = logical_screen_size
        self.orientation = orientation
        self.physical_frame_boundary = physical_frame_boundary


# =============================================================================
# Abstract base strategy
# =============================================================================

class ScalingStrategy(base_agent.EnvironmentInteractingAgent):
    """Base class that every test-time scaling strategy must subclass.

    Subclasses only need to implement `select_action`.  Everything else
    (environment interaction, SoM annotation, summarisation, history
    management) is handled here, mirroring M3A's logic.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "ScalingStrategy",
        wait_after_action_seconds: float = 2.0,
    ):
        super().__init__(env, name)
        self.llm = llm
        self.history: list[dict[str, Any]] = []
        self.additional_guidelines: list[str] | None = None
        self.wait_after_action_seconds = wait_after_action_seconds
        # Optional callback set by the benchmark runner for per-step logging.
        # Signature: (candidates: list[CandidateResult],
        #             aggregated: list[AggregatedCandidate]) -> None
        self._step_logger: Optional[
            Callable[[list[CandidateResult], list[AggregatedCandidate]], None]
        ] = None
        # Optional callback for summary-step logging.
        # Signature: (step_num, summary_prompt, summary_response) -> None
        self._summary_logger: Optional[
            Callable[[int, str, str], None]
        ] = None

    # ----- public helpers (same interface as M3A) -----

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.history = []

    # ----- abstract hook -----

    @abc.abstractmethod
    def select_action(
        self,
        ctx: StepContext,
    ) -> tuple[str, str]:
        """Pick an action given the current observation.

        Args:
            ctx: A StepContext with all information for this decision point.

        Returns:
            (reason, action_json_string)  — e.g.
            ("I need to tap the search bar", '{"action_type":"click","index":5}')
        """

    # ----- step logic (mirrors M3A) -----

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """One agent step: observe → select_action → execute → summarise."""
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

        # ---- 1. observe ----
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

        # ---- 2. select action via strategy ----
        reason, action = self.select_action(ctx)

        if (not reason) or (not action):
            logging.info("Strategy returned empty reason/action.")
            step_data["summary"] = (
                "Strategy did not return a valid action, so no action performed."
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        logging.info("Action: %s", action)
        logging.info("Reason: %s", reason)
        step_data["action_reason"] = reason
        step_data["action_output"] = f"Reason: {reason}\nAction: {action}"

        # ---- 3. convert & validate ----
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
            logging.info("Index %s out of range (%d elements).",
                         converted.index, len(before_ui_elements))
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

        # ---- 4. terminal actions ----
        if converted.action_type == "status":
            if converted.goal_status == "infeasible":
                logging.info("Agent thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted.action_type == "answer":
            logging.info("Agent answered: %s", converted.text)

        # ---- 5. execute ----
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

        # ---- 6. summarise (reuse LLM) ----
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

        m3a_utils.add_screenshot_label(step_data["before_screenshot_with_som"], "before")
        m3a_utils.add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = m3a._summarize_prompt(
            action, reason, goal,
            before_ui_elements_text, after_ui_text,
        )
        summary, is_safe, raw_resp = self.llm.predict_mm(
            summary_prompt,
            [step_data["before_screenshot_with_som"], after_screenshot],
        )

        if is_safe is False:
            summary = "Summary triggered LLM safety classifier."
        if not raw_resp:
            logging.info("Error calling LLM in summarisation phase: %s", summary)
            step_data["summary"] = f"Error in summarisation: {summary}"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        step_data["summary_raw_response"] = raw_resp
        logging.info("Summary: %s", summary)

        # Notify summary logger
        if self._summary_logger is not None:
            self._summary_logger(
                len(self.history) + 1, summary_prompt, summary,
            )

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
