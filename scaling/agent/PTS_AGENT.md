# Planner-Translator-Summarizer (PTS) Agent Architecture

> Created: 2026-02-16 | Version: Lc-v0.1.1

## Overview

A new three-phase agent architecture that decomposes each decision step into
**Planning**, **Translation**, and **Summarization** — each enhanced with
test-time scaling (best-of-N) for improved quality.

```
┌─────────────────────────────────────────────────────┐
│                   PTS Agent Step                    │
│                                                     │
│  ① Observe screen → StepContext                     │
│           ↓                                         │
│  ② Planner (best-of-N select)                       │
│     • Generate N candidate plans                    │
│     • Verifier selects best m plans                 │
│     • Considers previous plan + 3-retry limit       │
│           ↓                                         │
│  ③ Translator (best-of-N weighted)                  │
│     • Generate N candidate actions from plan        │
│     • Verifier scores each action (0-10)            │
│     • Group by action, sum scores, pick highest     │
│           ↓                                         │
│  ④ Execute action on device                         │
│           ↓                                         │
│  ⑤ Summarizer (best-of-N select)                    │
│     • Generate N candidate summaries                │
│     • Verifier selects the best summary             │
└─────────────────────────────────────────────────────┘
```

## Architecture Details

### 1. Planner 
(`scaling/agent/planner/planner.py`)

- **Method:** Best-of-N selection
- **Input:** Goal, action history, guidelines, previous plan, screenshots
- **Output:** A numbered step-by-step plan (1-5 steps)
- **Scaling:** Generator produces N plans → Verifier selects best m
- **Key feature:** Reviews previous plan against new situation; enforces
  a 3-retry limit to avoid repeating failing actions

### 2. Translator 
(`scaling/agent/translator/translator.py`)

- **Method:** Best-of-N weighted
- **Input:** Plan from planner, UI elements, screenshots
- **Output:** A single action JSON (no reason/explanation — reasoning is in the plan)
- **Scaling:** Generator produces N actions → Verifier scores each (0-10) →
  Actions grouped by string, scores summed → Highest total score wins

### 3. Summarizer 
(`scaling/agent/summarizer/summarizer.py`)

- **Method:** Best-of-N selection
- **Input:** Goal, action performed, before/after screenshots & UI elements
- **Output:** A concise summary (< 50 words)
- **Scaling:** Generator produces N summaries → Verifier selects best one

## File Structure

```
scaling/agent/
├── __init__.py                    # Package exports
├── prompts.py                     # All prompt templates
├── agent_strategy.py              # Main strategy (ties all three together)
├── planner/
│   ├── __init__.py
│   └── planner.py                 # Planner with best-of-N selection
├── translator/
│   ├── __init__.py
│   └── translator.py              # Translator with best-of-N weighted
└── summarizer/
    ├── __init__.py
    └── summarizer.py              # Summarizer with best-of-N selection
```

## Modified Files

| File | Change |
|------|--------|
| `scaling/__init__.py` | Added `PlannerTranslatorSummarizerStrategy` export |
| `config.yaml` | Added `pts_agent` section with all component settings |
| `client.py` (`AppConfig`) | Added PTS config fields + loading from YAML |
| `run_benchmark.py` | Added `pts_agent` strategy instantiation branch |

## Configuration

In `config.yaml`:

```yaml
scaling:
  strategy: "pts_agent"

  pts_agent:
    planner_n: 3              # N candidate plans
    planner_m: 1              # best m selected by verifier
    planner_temperature: 0.7

    translator_n: 3           # N candidate actions
    translator_temperature: 0.7

    summarizer_n: 3           # N candidate summaries
    summarizer_temperature: 0.7

    verifier_model: "Qwen3-VL-30B-A3B-Instruct"
    verifier_backend: "openai_compatible"
```

## Usage

```bash
# Run with PTS agent from config
python run_benchmark.py --scaling=pts_agent

# Override component settings via config.yaml
# (CLI overrides for PTS-specific settings can be added as needed)
```

## Design Decisions

1. **Shared verifier client:** All three components share one verifier
   client instance (lazy-initialized) to avoid redundant connections.

2. **Plan carries forward:** The planner stores `previous_plan` and
   passes it to the next step so the LLM can revise rather than
   re-plan from scratch.

3. **3-retry enforcement:** The planner prompt explicitly instructs the
   LLM not to repeat the same failing action more than 3 times, checking
   against the action history.

4. **Translator outputs action-only:** Unlike the baseline M3A which
   outputs `Reason: ... Action: ...`, the translator outputs only the
   action JSON. The plan text serves as the "reason" in logs and history.

5. **Async parallelism:** The translator scores N candidates in parallel
   (async), consistent with the existing `BestOfNWeightedStrategy`.
