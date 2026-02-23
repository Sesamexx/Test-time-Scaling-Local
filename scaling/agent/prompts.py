# =============================================================================
# scaling/agent/prompts.py
# Created:  [Lc-v0.1.1] 2026-02-15
# =============================================================================
"""
Prompt templates for the Planner-Translator-Summarizer agent.

All prompts are collected here for easy editing and review.
"""

# =============================================================================
# Valid actions reference (shared across prompts)
# =============================================================================

VALID_ACTIONS_JSON = """\
Valid action types:
  - If the current step is: Open an app <app_name>,
    then you should output: {"action_type": "open_app", "app_name": <app_name>}
  - If the current step is: Click on a UI element,
    then you should output: {"action_type": "click", "index": <target_index>}
  - If the current step is: Long press on a UI element,
    then you should output: {"action_type": "long_press", "index": <target_index>}
  - If the current step is: Input text into a UI element,
    then you should output: {"action_type": "input_text", "text": <text_input>, "index": <target_index>}
  - If the current step is: Scroll <up, down, left, right> on a UI element,
    then you should output: {"action_type": "scroll", "direction": <direction>, "index": <optional_target_index>}
  - If the current step is: Press the "Enter" key on the keyboard,
    then you should output: {"action_type": "keyboard_enter"}
  - If the current step is: Navigate to the home screen,
    then you should output: {"action_type": "navigate_home"}
  - If the current step is: Navigate back to the previous screen,
    then you should output: {"action_type": "navigate_back"}
  - If the current step is: Wait for a moment,
    then you should output: {"action_type": "wait"}
  - If the current step is: Provide an answer: <answer_text>,
    then you should output: {"action_type": "answer", "text": <answer_text>}
  - If the current step is: Mark task as complete,
    then you should output: {"action_type": "status", "goal_status": "complete"}
  - If the current step is: Mark task as infeasible,
    then you should output: {"action_type": "status", "goal_status": "infeasible"}
"""

VALID_ACTIONS_PLAIN = """\
Valid action types:
  1. Open an app <app_name>.
  2. Click on a UI element.
  3. Long press on a UI element.
  4. Input text into a UI element.
  5. Scroll <up, down, left, right> on a UI element.
  6. Press the "Enter" key on the keyboard.
  7. Navigate to the home screen.
  8. Navigate back to the previous screen.
  9. Wait for a moment.
  10. Provide an answer: <answer_text>
  11. Mark task as complete.
  12. Mark task as infeasible.
"""

# =============================================================================
# Planner prompts
# =============================================================================

PLANNER_PROMPT = """\
You are a planner for an Android phone agent. Your job is to produce a \
step-by-step plan of what to do next to achieve the user's goal.

## Goal
{goal}

## Available actions
{valid_actions}

## Action history
{history}

## Previous plan
{previous_plan}

## Guidelines
{guidelines}

## Instructions
You are given two screenshots of the current screen: the original and the \
same screen annotated with bounding boxes and numeric indexes on UI elements.

Review the previous plan (if any) against the current situation shown in \
the screenshots. Decide whether to continue, revise, or create a new plan.

**Important rules:**
- Do NOT repeat the same failing action more than 3 times. If something \
  has been tried 3 times in the history without success, try a different \
  approach.
- Keep the plan concise: 1-8 steps maximum.
- Each step should describe *what* to do, not the exact action JSON.
- If the goal is already complete, plan should be: "Mark task as complete."
- Your first step should be as specific as possible, choosing one from the \
  12 valid actions. Later steps can be your best guess.

## Output format
Reply with ONLY a numbered plan, one step per line. No other text.
Example:
1. Open the Settings app
2. Tap on "Wi-Fi"
3. Toggle the Wi-Fi switch on
4. Mark task as complete

Your plan:
"""

PLANNER_VERIFIER_PROMPT = """\
You are a judge evaluating plans for an Android phone agent.

## Goal
{goal}

## Action history
{history}

## Candidate plans to evaluate
{candidates}

## Valid actions
{valid_actions}

## Instructions
You are given two screenshots of the current screen.

Select the BEST plan(s) from the candidates above. A good plan:
1. Is relevant to the current screen state shown in the screenshots.
2. Makes logical progress toward the goal given the history.
3. Does NOT repeat actions that have already failed multiple times.
4. Is concise and actionable.
5. The first step should be as specific as possible, choosing one from the 12 valid actions.

## Output format
Reply with ONLY the number(s) of the best plan(s), comma-separated. \
Pick at most {m} plans.
Example: 1

Your selection:
"""


# =============================================================================
# Translator prompts
# =============================================================================

TRANSLATOR_PROMPT = """\
You are an actor for an Android phone agent. Given a step and the \
current screen, output the NEXT single action to execute.

## CRITICAL: Output format
Your output must be a **single JSON object** starting with {{ and ending with }}.
Do NOT output anything else. No text, no explanation, no "action:" prefix.

## CRITICAL: System actions (NOT UI interactions!)
These steps map to special system actions. Do NOT use "click" for them:
- "Mark task as complete" → {{"action_type": "status", "goal_status": "complete"}}
- "Mark task as infeasible" → {{"action_type": "status", "goal_status": "infeasible"}}
- "Provide an answer: <text>" → {{"action_type": "answer", "text": "<text>"}}

If the current step is one of the above, output the corresponding JSON \
and NOTHING else. Do NOT click any UI element.

## Rules for UI actions
- You MUST use "index" (integer from the UI element list) for click, \
  long_press, input_text, scroll. NEVER use "point", "element_index", \
  or coordinate values.
- Do NOT add extra keys like "explain" — output only the required keys.

## Examples of CORRECT output:
{{"action_type": "click", "index": 1}}
{{"action_type": "open_app", "app_name": "Audio Recorder"}}
{{"action_type": "input_text", "text": "hello", "index": 3}}
{{"action_type": "scroll", "direction": "down"}}
{{"action_type": "status", "goal_status": "complete"}}
{{"action_type": "navigate_home"}}

## Examples of WRONG output (DO NOT output like these):
action:open_app app_name:Audio Recorder     -> WRONG! Not JSON!
action:CLICK point:500,897                  -> WRONG! Not JSON, uses "point"!
{{"action_type": "click", "point": [500, 897]}} -> WRONG! Must use "index", not "point"!

## UI elements
{ui_elements}

## Current step
{plan}

## Available actions:
{valid_actions}

Your action (JSON only):
"""

TRANSLATOR_VERIFIER_PROMPT = """\
Score this Android agent action from 1 to 10.

Current step: {plan}
Candidate action: {action}

You are given two screenshots of the current screen with UI element indexes.

## Scoring rules
- If the action does NOT target a specific UI element (e.g. open_app, \
navigate_home, navigate_back, keyboard_enter, wait, status, answer), \
just check: does the action make sense for the current step? \
If yes → 10, if no → 1.
- If the action targets a UI element by index (click, long_press, \
input_text, scroll), check: does the index correspond to the correct \
element for the current step as shown in the annotated screenshot? \
Perfect match = 10, reasonable = 7, wrong target = 3, nonsensical = 1.

Reply with ONLY a single integer 1-10. Nothing else.
"""


# =============================================================================
# Summarizer prompts
# =============================================================================

SUMMARIZER_PROMPT = """\
You are summarizing the latest step of an Android phone agent.

## Goal
{goal}

## Action performed
{action}

## Reason / plan context
{reason}

## UI elements before action
{before_elements}

## UI elements after action
{after_elements}

## Instructions
You are given two screenshots: "before" (with label) and "after" (with label).

Compare the two screenshots and the UI element lists. Write a brief \
summary of what happened. Include:
- What was intended
- Whether it worked as expected

Keep it under 50 words, single line.

Summary:
"""

SUMMARIZER_VERIFIER_PROMPT = """\
You are judging summaries of an Android phone agent step.

## Goal
{goal}

## Action performed
{action}

## Candidate summaries
{candidates}

## Instructions
You are given the "before" and "after" screenshots.

Select the BEST summary. A good summary:
1. Accurately describes what happened on screen.
2. Is concise (under 50 words).
3. Includes useful information for future decision-making.

## Output format
Reply with ONLY the number of the best summary.
Example: 2

Your selection:
"""
