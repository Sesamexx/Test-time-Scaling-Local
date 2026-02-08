# =============================================================================
# run_benchmark.py
# Created:  [Lc-v0.0.2] 2026-02-04
# Updated:  [Lc-v0.0.2] 2026-02-04 Add clean logging
# =============================================================================
"""
Run AndroidWorld benchmark with gelab-zero-4b-preview model.

Usage:
    # Run full benchmark
    python run_benchmark.py
    
    # Run specific task
    python run_benchmark.py --tasks=ContactsAddContact
    
    # Run with emulator setup (first time only)
    python run_benchmark.py --perform_emulator_setup
"""

import os
import sys
import asyncio
import json
from typing import Any, Optional

import numpy as np

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
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.env import env_launcher

from client import get_client, Config

logging.set_verbosity(logging.INFO)

# =============================================================================
# Flags
# =============================================================================

_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    os.path.expandvars(r'%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe'),
    'Path to adb executable.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup (first time only).',
)
_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device.',
)
_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run. If None, run all tasks.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'Directory to save checkpoints and resume from.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'Path to save results.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed',
    30,
    'Random seed for task randomness.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'gelab-zero-4b-preview',
    'Model name to use for inference.',
)
_TEMPERATURE = flags.DEFINE_float(
    'temperature',
    0.0,
    'Sampling temperature for the model.',
)

# =============================================================================
# GelabWrapper - Multimodal LLM Wrapper for Remote Server
# =============================================================================

class SimpleLogger:
    """Simple logger that writes clean, human-readable logs."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.step_count = 0
        
        # Create log file with header
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AndroidWorld Benchmark - Clean Log\n")
            f.write(f"Model: gelab-zero-4b-preview\n")
            f.write(f"Started: {self._get_timestamp()}\n")
            f.write("=" * 80 + "\n\n")
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log_task_start(self, task_name: str, goal: str):
        """Log task start."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TASK: {task_name}\n")
            f.write(f"GOAL: {goal}\n")
            f.write(f"TIME: {self._get_timestamp()}\n")
            f.write("=" * 80 + "\n\n")
        self.step_count = 0
    
    def log_step(self, request_summary: str, response: str, action: str):
        """Log a single step."""
        self.step_count += 1
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"--- Step {self.step_count} ---\n")
            f.write(f"Request: {request_summary}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Action: {action}\n\n")
    
    def log_task_result(self, task_name: str, success: bool, details: str = ""):
        """Log task completion."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'✓' if success else '✗'} RESULT: {task_name} - ")
            f.write(f"{'SUCCESS' if success else 'FAILED'}\n")
            if details:
                f.write(f"Details: {details}\n")
            f.write("\n")


class GelabWrapper(infer.MultimodalLlmWrapper):
    """Wrapper for gelab-zero-4b-preview model via remote server."""
    
    def __init__(
        self,
        model_name: str = "gelab-zero-4b-preview",
        config_path: str = "config.yaml",
        max_retry: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        simple_logger: Optional[SimpleLogger] = None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            model_name: Model name on the server.
            config_path: Path to config.yaml with server URL.
            max_retry: Maximum retry attempts on failure.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            simple_logger: Optional simple logger for clean output.
        """
        self.model_name = model_name
        self.config_path = config_path
        self.max_retry = max_retry
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.simple_logger = simple_logger
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
        
        # Log request to simple logger
        if self.simple_logger:
            num_images = len(images) if images else 0
            request_summary = f"{num_images} image(s), prompt length: {len(text_prompt)} chars"
        
        for attempt in range(self.max_retry):
            try:
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    images=images if images else None
                )
                
                # Log response to simple logger
                if self.simple_logger:
                    try:
                        # Try to extract JSON using the same method as M3A
                        from android_world.agents import agent_utils
                        extracted_json = agent_utils.extract_json(response)
                        if extracted_json:
                            action_type = extracted_json.get("action_type", "unknown")
                            action_str = f"Action: {action_type}"
                            if "index" in extracted_json:
                                action_str += f" (index: {extracted_json['index']})"
                            if "text" in extracted_json:
                                action_str += f" (text: {extracted_json['text'][:30]}...)"
                        else:
                            action_str = "Failed to extract JSON"
                    except Exception as e:
                        action_str = f"Parse error: {str(e)[:50]}"
                    
                    self.simple_logger.log_step(
                        request_summary=request_summary,
                        response=response,
                        action=action_str
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
        
        # Log failure
        if self.simple_logger:
            self.simple_logger.log_step(
                request_summary=request_summary,
                response=f"ERROR: {last_error}",
                action="FAILED"
            )
        
        return infer.ERROR_CALLING_LLM, None, None

# =============================================================================
# Main
# =============================================================================

class LoggingM3A(m3a.M3A):
    """M3A agent with simple logging that tracks task starts."""
    
    def __init__(self, env, llm, simple_logger: Optional[SimpleLogger] = None, **kwargs):
        super().__init__(env, llm, **kwargs)
        self.simple_logger = simple_logger
        self.current_goal = None
        self.step_counter = 0
    
    def step(self, goal: str):
        """Override step to log new tasks."""
        if self.simple_logger and goal != self.current_goal:
            # New goal detected - log task start
            self.current_goal = goal
            self.step_counter = 0
            # We don't have the task name here, so just use "Task"
            self.simple_logger.log_task_start("Current Task", goal)
        
        self.step_counter += 1
        result = super().step(goal)
        return result





def _main() -> None:
    """Run the benchmark."""
    print("=" * 60)
    print("AndroidWorld Benchmark - gelab-zero-4b-preview")
    print("=" * 60)
    
    # Load and setup environment
    print("\n[1/5] Setting up Android environment...")
    env = env_launcher.load_and_setup_env(
        console_port=_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    print("  ✓ Environment ready")
    
    # Create task suite
    print("\n[2/5] Creating task suite...")
    task_registry = registry.TaskRegistry()
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY),
        n_task_combinations=_N_TASK_COMBINATIONS.value,
        seed=_TASK_RANDOM_SEED.value,
        tasks=_TASKS.value,
    )
    suite.suite_family = registry.TaskRegistry.ANDROID_WORLD_FAMILY
    print(f"  ✓ Suite created with {len(suite)} task types")
    
    # Setup checkpoint directory
    if _CHECKPOINT_DIR.value:
        checkpoint_dir = _CHECKPOINT_DIR.value
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
    
    # Create simple logger
    print("\n[3/5] Setting up logging...")
    simple_log_path = checkpoint_dir + "_simple.log"
    os.makedirs(os.path.dirname(checkpoint_dir) if os.path.dirname(checkpoint_dir) else ".", exist_ok=True)
    simple_logger = SimpleLogger(simple_log_path)
    print(f"  ✓ Clean log: {simple_log_path}")
    print(f"  ✓ Detailed log: {checkpoint_dir}")
    
    # Create agent with Gelab wrapper
    print("\n[4/5] Initializing agent...")
    llm_wrapper = GelabWrapper(
        model_name=_MODEL_NAME.value,
        temperature=_TEMPERATURE.value,
        simple_logger=simple_logger,
    )
    agent = LoggingM3A(env, llm_wrapper, simple_logger=simple_logger)
    agent.name = f"m3a_{_MODEL_NAME.value}"
    agent.transition_pause = None  # Use auto mode
    print(f"  ✓ Agent ready: {agent.name}")
    
    print(f"\n[5/5] Running benchmark...")
    print(f"  Output: {checkpoint_dir}")
    
    # Run benchmark
    try:
        results = suite_utils.run(
            suite,
            agent,
            checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
            demo_mode=False,
        )
        print("\n" + "=" * 60)
        print("Benchmark completed successfully!")
        print(f"Detailed results: {checkpoint_dir}")
        print(f"Clean log: {simple_log_path}")
        print("=" * 60)
    finally:
        env.close()


def main(argv):
    del argv
    _main()


if __name__ == "__main__":
    app.run(main)
