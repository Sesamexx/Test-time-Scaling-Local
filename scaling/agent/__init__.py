# =============================================================================
# scaling/agent/__init__.py
# Created:  [Lc-v0.1.1] 2026-02-16
# =============================================================================
"""
Planner-Translator-Summarizer agent architecture.

A three-phase agent that decomposes each step into:
  1. Planner   — best-of-N selection: generate N plans, verifier picks best m.
  2. Translator — best-of-N weighted: generate N actions, pick highest total score.
  3. Summarizer — best-of-N selection: generate N summaries, verifier picks best.
"""

from scaling.agent.agent_strategy import PlannerTranslatorSummarizerStrategy

__all__ = ["PlannerTranslatorSummarizerStrategy"]
