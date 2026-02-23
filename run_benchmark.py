# =============================================================================
# run_benchmark.py
# Created:  [Lc-v0.0.2] 2026-02-04
# Updated:  [Lc-v0.0.4] 2026-02-16 Add scaling strategies
# =============================================================================
"""
Run AndroidWorld benchmark with gelab-zero-4b-preview model.

All defaults live in config.yaml.  CLI flags override them when provided.

Usage:
    # Run with defaults from config.yaml
    python run_benchmark.py

    # Override scaling strategy on CLI
    python run_benchmark.py --scaling=best_of_n_weighted --n_samples=16

    # Run specific task
    python run_benchmark.py --tasks=ContactsAddContact

    # Run with emulator setup (first time only)
    python run_benchmark.py --perform_emulator_setup
"""

import os
import sys
import asyncio
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Force UTF-8 on Windows
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

# Disable proxies for local server
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# Suppress gRPC verbosity
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'none'

from absl import app
from absl import flags
from absl import logging

# Add android_world to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "android_world"))

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import infer
from android_world.env import env_launcher

from client import get_client, AppConfig

# Scaling strategies
from scaling import BaselineStrategy, BestOfNWeightedStrategy, PlannerTranslatorSummarizerStrategy

logging.set_verbosity(logging.INFO)

# =============================================================================
# Flags (optional CLI overrides)
# =============================================================================

_CONFIG_PATH = flags.DEFINE_string(
    'config', 'config.yaml',
    'Path to the YAML configuration file.',
)
_ADB_PATH = flags.DEFINE_string(
    'adb_path', None,
    'Override: path to adb executable.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup', None,
    'Override: whether to perform emulator setup (first time only).',
)
_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port', None,
    'Override: console port of the running Android device.',
)
_TASKS = flags.DEFINE_list(
    'tasks', None,
    'Override: specific tasks to run (comma-separated). If None, use config.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations', None,
    'Override: number of task instances per template.',
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir', None,
    'Override: directory to save/resume checkpoints.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None,
    'Override: path to save results.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', None,
    'Override: random seed for task randomness.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', None,
    'Override: model name for inference.',
)
_TEMPERATURE = flags.DEFINE_float(
    'temperature', None,
    'Override: sampling temperature for the model.',
)
_SCALING = flags.DEFINE_string(
    'scaling', None,
    'Override: scaling strategy ("baseline" | "best_of_n_weighted").',
)
_N_SAMPLES = flags.DEFINE_integer(
    'n_samples', None,
    'Override: number of candidate samples for best-of-N strategies.',
)
_ACTOR_TEMPERATURE = flags.DEFINE_float(
    'actor_temperature', None,
    'Override: actor sampling temperature for best-of-N.',
)
_VERIFIER_MODEL = flags.DEFINE_string(
    'verifier_model', None,
    'Override: verifier model name for best-of-N weighted.',
)
_VERIFIER_BACKEND = flags.DEFINE_string(
    'verifier_backend', None,
    'Override: verifier backend ("local" | "openai_compatible").',
)
_PLANNER_N = flags.DEFINE_integer(
    'planner_n', None,
    'Override: number of candidate plans for PTS planner.',
)
_TRANSLATOR_N = flags.DEFINE_integer(
    'translator_n', None,
    'Override: number of candidate actions for PTS translator.',
)
_SUMMARIZER_N = flags.DEFINE_integer(
    'summarizer_n', None,
    'Override: number of candidate summaries for PTS summarizer.',
)


def _resolve_config() -> AppConfig:
    """Load config.yaml, then overlay any CLI flag overrides."""
    cfg = AppConfig.load(_CONFIG_PATH.value)

    if _ADB_PATH.value is not None:
        cfg.adb_path = _ADB_PATH.value
    if _EMULATOR_SETUP.value is not None:
        cfg.perform_emulator_setup = _EMULATOR_SETUP.value
    if _CONSOLE_PORT.value is not None:
        cfg.console_port = _CONSOLE_PORT.value
    if _TASKS.value is not None:
        cfg.tasks = _TASKS.value
    if _N_TASK_COMBINATIONS.value is not None:
        cfg.n_task_combinations = _N_TASK_COMBINATIONS.value
    if _CHECKPOINT_DIR.value is not None:
        cfg.checkpoint_dir = _CHECKPOINT_DIR.value
    if _OUTPUT_PATH.value is not None:
        cfg.output_path = _OUTPUT_PATH.value
    if _TASK_RANDOM_SEED.value is not None:
        cfg.task_random_seed = _TASK_RANDOM_SEED.value
    if _MODEL_NAME.value is not None:
        cfg.model_name = _MODEL_NAME.value
    if _TEMPERATURE.value is not None:
        cfg.model_temperature = _TEMPERATURE.value
    if _SCALING.value is not None:
        cfg.scaling_strategy = _SCALING.value
    if _N_SAMPLES.value is not None:
        cfg.bon_n_samples = _N_SAMPLES.value
    if _ACTOR_TEMPERATURE.value is not None:
        cfg.bon_actor_temperature = _ACTOR_TEMPERATURE.value
    if _VERIFIER_MODEL.value is not None:
        cfg.bon_verifier_model = _VERIFIER_MODEL.value
    if _VERIFIER_BACKEND.value is not None:
        cfg.bon_verifier_backend = _VERIFIER_BACKEND.value
    if _PLANNER_N.value is not None:
        cfg.pts_planner_n = _PLANNER_N.value
    if _TRANSLATOR_N.value is not None:
        cfg.pts_translator_n = _TRANSLATOR_N.value
    if _SUMMARIZER_N.value is not None:
        cfg.pts_summarizer_n = _SUMMARIZER_N.value

    return cfg

# =============================================================================
# GelabWrapper - Multimodal LLM Wrapper for Remote Server
# =============================================================================

class SimpleLogger:
    """Two-level human-readable logger (info + debug).

    * info log  - all candidates with their scores, aggregated ranking,
      and summarisation results.
    * debug log - everything in info PLUS full actor prompts, raw LLM
      responses, and verifier raw responses.

    Both files are plain-text; the info log is also echoed to stdout.

    Step numbering is tracked internally and reset on each new task.
    """

    def __init__(self, base_path: str, model_name: str = "gelab-zero-4b-preview"):
        self.model_name = model_name
        self.info_path = base_path + ".info.log"
        self.debug_path = base_path + ".debug.log"
        self._step = 0  # reset per task

        sep = "=" * 80
        header = (
            sep + "\n"
            "AndroidWorld Benchmark Log\n"
            f"Model : {self.model_name}\n"
            f"Start : {self._ts()}\n"
            + sep + "\n\n"
        )
        for p in (self.info_path, self.debug_path):
            with open(p, "w", encoding="utf-8") as f:
                f.write(header)

    # ---- helpers ----

    @staticmethod
    def _ts():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level: str, text: str):
        """Append *text* to the appropriate log file(s).

        level="info"  → write to both info and debug logs.
        level="debug" → write to debug log only.
        """
        if level == "info":
            for p in (self.info_path, self.debug_path):
                with open(p, "a", encoding="utf-8") as f:
                    f.write(text)
        else:
            with open(self.debug_path, "a", encoding="utf-8") as f:
                f.write(text)

    # ---- task-level callbacks (wired to suite_utils) ----

    def log_task_start(self, task_name: str, goal: str):
        """Called before a task episode begins."""
        self._step = 0
        sep = "=" * 80
        block = (
            "\n" + sep + "\n"
            f"TASK : {task_name}\n"
            f"GOAL : {goal}\n"
            f"TIME : {self._ts()}\n"
            + sep + "\n\n"
        )
        self._write("info", block)
        print(f"\n>>> Task: {task_name}")
        print(f"    Goal: {goal}")

    def log_task_end(self, task_name: str, goal: str, reward: float,
                     n_correct: int, n_total: int):
        """Called after a task episode finishes."""
        import math
        if math.isnan(reward):
            sym, label = "⚠", "ERROR"
        elif reward >= 1.0:
            sym, label = "✓", "SUCCESS"
        else:
            sym, label = "✗", "FAILED"

        pct = (n_correct / n_total * 100) if n_total > 0 else 0.0
        block = (
            f"\n{sym} RESULT: {task_name} — {label}\n"
            f"  Overall so far: {n_correct}/{n_total} ({pct:.1f}%)\n"
            + "-" * 80 + "\n"
        )
        self._write("info", block)
        print(f"  {sym} {task_name}: {label}  "
              f"[overall {n_correct}/{n_total} = {pct:.1f}%]")

    def log_summary(self, n_correct: int, n_total: int):
        """Final summary block at the very end."""
        pct = (n_correct / n_total * 100) if n_total > 0 else 0.0
        sep = "=" * 80
        block = (
            "\n" + sep + "\n"
            "FINAL SUMMARY\n"
            f"  Score : {n_correct}/{n_total} ({pct:.1f}%)\n"
            f"  Time  : {self._ts()}\n"
            + sep + "\n"
        )
        self._write("info", block)

    # ---- step-level callback (wired to ScalingStrategy._step_logger) ----

    def log_step_candidates(self, candidates, aggregated):
        """Called by the scaling strategy after scoring candidates.

        Args:
            candidates: list[CandidateResult] sorted by score descending.
            aggregated: list[AggregatedCandidate] sorted by total_score desc.
        """
        import math

        if not candidates:
            self._write("info", f"  [Translator] (no candidates)\n")
            return

        n = len(candidates)

        # =====================================================================
        # INFO level — all candidates + verifier response + aggregated ranking
        # =====================================================================
        info_lines = [f"  [Translator] Candidates ({n}):\n"]

        # Individual candidates
        for i, c in enumerate(candidates):
            s = f"{c.score:.0f}" if not math.isnan(c.score) else "n/a"
            info_lines.append(
                f"    #{i+1}  score={s}  | {c.action}\n"
            )
            # Show verifier response for every candidate
            if c.verifier_response:
                info_lines.append(
                    f"         verifier: {c.verifier_response}\n"
                )

        # Aggregated ranking
        if len(aggregated) > 0 and len(candidates) > 1:
            info_lines.append(f"  [Translator] Aggregated ranking ({len(aggregated)} unique actions):\n")
            for i, a in enumerate(aggregated):
                ts_str = f"{a.total_score:.0f}" if not math.isnan(a.total_score) else "n/a"
                info_lines.append(
                    f"    #{i+1}  total={ts_str} (×{a.count})  | {a.action}\n"
                )

        # Chosen action
        best = aggregated[0] if aggregated else None
        if best:
            bs = f"{best.total_score:.0f}" if not math.isnan(best.total_score) else "n/a"
            info_lines.append(f"  [Translator] → Selected: total={bs} (×{best.count}) | {best.action}\n")

        self._write("info", "".join(info_lines))

        # Print a concise version to stdout
        if best:
            bs = f"{best.total_score:.0f}" if not math.isnan(best.total_score) else "n/a"
            print(f"  Step {self._step} | total={bs} (×{best.count}) | {best.action}")

        # =====================================================================
        # DEBUG level — add full prompts & raw responses
        # =====================================================================
        dbg_lines = []
        for i, c in enumerate(candidates):
            s = f"{c.score:.0f}" if not math.isnan(c.score) else "n/a"
            dbg_lines.append(f"\n  ·· Translator Candidate {i+1}/{n} detail ··\n")
            if c.actor_prompt:
                dbg_lines.append(f"  [Actor Prompt]\n{c.actor_prompt}\n")
            if c.raw_response:
                dbg_lines.append(f"  [Actor Raw Response]\n{c.raw_response}\n")
            if c.verifier_response:
                dbg_lines.append(f"  [Verifier Raw Response]\n{c.verifier_response}\n")
        dbg_lines.append("\n")
        self._write("debug", "".join(dbg_lines))

    # ---- summary-step callback (wired to ScalingStrategy._summary_logger) --

    def log_summary_step(self, step_num: int, prompt: str, response: str):
        """Called after each step's summarisation LLM call.

        Args:
            step_num: The 1-based step number.
            prompt:   The summarisation prompt sent to the LLM.
            response: The summarisation response from the LLM.
        """
        # Info: just the summary result (the detailed candidates are
        # logged by log_summarizer_candidates)
        info_block = f"  [Summary] Final: {response}\n"
        self._write("info", info_block)

        # Debug: full prompt + result
        dbg_block = (
            f"\n  ·· Summarisation detail ··\n"
            f"  [Summary Prompt]\n{prompt}\n"
            f"  [Summary Response]\n{response}\n\n"
        )
        self._write("debug", dbg_block)

    # ---- planner-phase callback (wired to ScalingStrategy._planner_logger) --

    def log_planner_candidates(self, candidates, selected_indices, verifier_response):
        """Called by the planner after selecting best plan(s).

        Args:
            candidates:        list[str] — all candidate plans generated.
            selected_indices:  list[int] — 0-based indices of selected plans.
            verifier_response: str — raw verifier response text.
        """
        self._step += 1
        n = len(candidates)
        ts = self._ts()

        info_lines = [f"\n--- Step {self._step}  [{ts}] ---\n"]
        info_lines.append(f"  [Planner] Candidates ({n}):\n")
        for i, plan in enumerate(candidates):
            marker = " ★" if i in selected_indices else ""
            info_lines.append(f"    Plan #{i+1}{marker}:\n")
            for line in plan.splitlines():
                info_lines.append(f"      {line}\n")
        info_lines.append(f"  [Planner] Verifier response: {verifier_response}\n")
        sel_nums = ", ".join(str(i + 1) for i in selected_indices)
        info_lines.append(f"  [Planner] → Selected plan(s): {sel_nums}\n")

        self._write("info", "".join(info_lines))

    # ---- summarizer-phase callback (wired to ScalingStrategy._summarizer_logger) --

    def log_summarizer_candidates(self, candidates, selected_index, verifier_response):
        """Called by the summarizer after selecting best summary.

        Args:
            candidates:        list[str] — all candidate summaries generated.
            selected_index:    int — 0-based index of the selected summary.
            verifier_response: str — raw verifier response text.
        """
        n = len(candidates)

        info_lines = [f"  [Summarizer] Candidates ({n}):\n"]
        for i, summary in enumerate(candidates):
            marker = " ★" if i == selected_index else ""
            info_lines.append(f"    Summary #{i+1}{marker}: {summary}\n")
        info_lines.append(f"  [Summarizer] Verifier response: {verifier_response}\n")
        info_lines.append(f"  [Summarizer] → Selected summary: #{selected_index + 1}\n")

        self._write("info", "".join(info_lines))


class GelabWrapper(infer.MultimodalLlmWrapper):
    """Wrapper for gelab-zero-4b-preview model via remote server."""
    
    def __init__(
        self,
        model_name: str = "gelab-zero-4b-preview",
        config_path: str = "config.yaml",
        max_retry: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """
        Initialize the wrapper.  All defaults come from config.yaml via
        the caller; the parameters here are the *resolved* values.
        
        Args:
            model_name: Model name on the server.
            config_path: Path to config.yaml (used only for server URL).
            max_retry: Maximum retry attempts on failure.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.model_name = model_name
        self.config_path = config_path
        self.max_retry = max_retry
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of client."""
        if self._client is None:
            self._client = get_client(self.config_path)
        return self._client
    
    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        """Text-only prediction (calls predict_mm with no images)."""
        return self.predict_mm(text_prompt, [])
    
    def predict_mm(
        self,
        text_prompt: str,
        images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """
        Multimodal prediction with text and images.
        
        Args:
            text_prompt: Text prompt to send.
            images: List of images as numpy arrays.
            
        Returns:
            Tuple of (response_text, is_safe, raw_response).
        """
        # Handle event loop - create new one if none exists or current is closed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._predict_mm_async(text_prompt, images)
        )
    
    async def _predict_mm_async(
        self,
        text_prompt: str,
        images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Async implementation of predict_mm."""
        import time
        
        retry_delay = 1.0
        last_error = None
        
        for attempt in range(self.max_retry):
            try:
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    images=images if images else None
                )
                
                # Return (text, is_safe, raw_response)
                return response, True, {"response": response}
            
            except Exception as e:
                last_error = e
                print(f"[GelabWrapper] Attempt {attempt + 1}/{self.max_retry} failed: {e}")
                if attempt < self.max_retry - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        # All retries failed
        print(f"[GelabWrapper] All retries exhausted. Last error: {last_error}")
        
        return infer.ERROR_CALLING_LLM, None, None

# =============================================================================
# Main
# =============================================================================


def _main() -> None:
    """Run the benchmark."""
    # ---- resolve config (YAML + CLI overrides) ----
    cfg = _resolve_config()

    print("=" * 60)
    print(f"AndroidWorld Benchmark - {cfg.model_name}")
    print(f"  scaling : {cfg.scaling_strategy}")
    if cfg.scaling_strategy.lower() == "best_of_n_weighted":
        print(f"  verifier: {cfg.bon_verifier_model} ({cfg.bon_verifier_backend})")
    elif cfg.scaling_strategy.lower() == "pts_agent":
        print(f"  verifier: {cfg.pts_verifier_model} ({cfg.pts_verifier_backend})")
        print(f"  planner : N={cfg.pts_planner_n}, m={cfg.pts_planner_m}")
        print(f"  translator: N={cfg.pts_translator_n}")
        print(f"  summarizer: N={cfg.pts_summarizer_n}")
    print("=" * 60)
    
    # Load and setup environment
    print("\n[1/5] Setting up Android environment...")
    env = env_launcher.load_and_setup_env(
        console_port=cfg.console_port,
        emulator_setup=cfg.perform_emulator_setup,
        adb_path=cfg.adb_path,
    )
    print("  ✓ Environment ready")
    
    # Create task suite
    print("\n[2/5] Creating task suite...")
    task_registry = registry.TaskRegistry()
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY),
        n_task_combinations=cfg.n_task_combinations,
        seed=cfg.task_random_seed,
        tasks=cfg.tasks,
    )
    suite.suite_family = registry.TaskRegistry.ANDROID_WORLD_FAMILY
    print(f"  ✓ Suite created with {len(suite)} task types")
    
    # Setup checkpoint directory
    if cfg.checkpoint_dir:
        checkpoint_dir = cfg.checkpoint_dir
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(cfg.output_path)
    
    # Create simple logger
    print("\n[3/5] Setting up logging...")
    simple_log_base = checkpoint_dir + "_simple"
    os.makedirs(os.path.dirname(checkpoint_dir) if os.path.dirname(checkpoint_dir) else ".", exist_ok=True)
    simple_logger = SimpleLogger(simple_log_base, model_name=cfg.model_name)
    print(f"  ✓ Info  log: {simple_logger.info_path}")
    print(f"  ✓ Debug log: {simple_logger.debug_path}")
    print(f"  ✓ Checkpoint: {checkpoint_dir}")
    
    # Create agent with Gelab wrapper + scaling strategy
    print("\n[4/5] Initializing agent...")
    llm_wrapper = GelabWrapper(
        model_name=cfg.model_name,
        temperature=cfg.model_temperature,
        max_tokens=cfg.model_max_tokens,
        max_retry=cfg.model_max_retry,
    )

    # Build the appropriate scaling strategy
    scaling_name = cfg.scaling_strategy.lower()
    if scaling_name == "baseline":
        agent = BaselineStrategy(env, llm_wrapper)
    elif scaling_name == "best_of_n_weighted":
        agent = BestOfNWeightedStrategy(
            env,
            actor_llm=llm_wrapper,
            n_samples=cfg.bon_n_samples,
            actor_temperature=cfg.bon_actor_temperature,
            verifier_model=cfg.bon_verifier_model,
            verifier_backend=cfg.bon_verifier_backend,
            qwen_api_key=cfg.qwen_api_key,
            qwen_base_url=cfg.qwen_base_url,
        )
    elif scaling_name == "pts_agent":
        agent = PlannerTranslatorSummarizerStrategy(
            env,
            actor_llm=llm_wrapper,
            planner_n=cfg.pts_planner_n,
            planner_m=cfg.pts_planner_m,
            planner_temperature=cfg.pts_planner_temperature,
            translator_n=cfg.pts_translator_n,
            translator_temperature=cfg.pts_translator_temperature,
            summarizer_n=cfg.pts_summarizer_n,
            summarizer_temperature=cfg.pts_summarizer_temperature,
            verifier_model=cfg.pts_verifier_model,
            verifier_backend=cfg.pts_verifier_backend,
            qwen_api_key=cfg.qwen_api_key,
            qwen_base_url=cfg.qwen_base_url,
        )
    else:
        raise ValueError(f"Unknown scaling strategy: {scaling_name}")

    agent.name = f"{scaling_name}_{cfg.model_name}"
    agent.transition_pause = None  # Use auto mode
    # Wire step-level logger
    agent._step_logger = simple_logger.log_step_candidates
    # Wire summary-step logger
    agent._summary_logger = simple_logger.log_summary_step
    # Wire planner-phase logger
    agent._planner_logger = simple_logger.log_planner_candidates
    # Wire summarizer-phase logger
    agent._summarizer_logger = simple_logger.log_summarizer_candidates
    print(f"  ✓ Strategy: {scaling_name}")
    print(f"  ✓ Agent ready: {agent.name}")
    
    print(f"\n[5/5] Running benchmark...")
    print(f"  Output: {checkpoint_dir}")
    
    # Track running totals for the final summary
    _final_correct = 0
    _final_total = 0

    def _on_task_start(task_name: str, goal: str):
        simple_logger.log_task_start(task_name, goal)

    def _on_task_end(task_name: str, goal: str, reward: float,
                     n_correct: int, n_total: int):
        nonlocal _final_correct, _final_total
        _final_correct = n_correct
        _final_total = n_total
        simple_logger.log_task_end(task_name, goal, reward,
                                   n_correct, n_total)

    # Run benchmark
    try:
        results = suite_utils.run(
            suite,
            agent,
            checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
            demo_mode=False,
            on_task_start=_on_task_start,
            on_task_end=_on_task_end,
        )
        simple_logger.log_summary(_final_correct, _final_total)
        print("\n" + "=" * 60)
        print("Benchmark completed successfully!")
        pct = (_final_correct / _final_total * 100) if _final_total > 0 else 0.0
        print(f"  Final score: {_final_correct}/{_final_total} ({pct:.1f}%)")
        print(f"Detailed results: {checkpoint_dir}")
        print(f"Info  log: {simple_logger.info_path}")
        print(f"Debug log: {simple_logger.debug_path}")
        print("=" * 60)
    finally:
        env.close()


def main(argv):
    del argv
    _main()


if __name__ == "__main__":
    app.run(main)
