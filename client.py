# =============================================================================
# client.py
# Created:  [Lc-v0.0.1] 2026-02-03
# Updated:  [Lc-v0.0.3] 2026-02-12 Add OpenAI-compatible verifier backend
# =============================================================================
"""
Client for remote LLM inference server.
Supports both text-only and multimodal (image) requests.
Also exposes the full config.yaml through AppConfig.
"""

import yaml
import httpx
import base64
import io
import os
from typing import List, Optional, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Load .env file if present (for QWEN_API_KEY / QWEN_BASE_URL)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed — fall back to manual .env parsing
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_env_path):
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    os.environ.setdefault(_key.strip(), _val.strip())

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Server-only config (backward compatible)."""
    server_url: str
    timeout: float = 300.0
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        """Load server config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            server_url=data["server"]["url"],
            timeout=data["server"].get("timeout", 300.0)
        )


@dataclass
class AppConfig:
    """Full application config — single source of truth for all settings.

    Mirrors every section in config.yaml *except* server connection
    (which is handled by the legacy ``Config`` class used by ``get_client``).

    Call ``AppConfig.load()`` once at startup and pass the instance around.
    """

    # -- actor model --
    model_name: str = "gelab-zero-4b-preview"
    model_temperature: float = 0.0
    model_max_tokens: int = 2048
    model_max_retry: int = 3

    # -- benchmark --
    adb_path: str = ""
    console_port: int = 5554
    perform_emulator_setup: bool = False
    n_task_combinations: int = 1
    task_random_seed: int = 30
    output_path: str = "~/android_world/runs"
    checkpoint_dir: str = ""
    tasks: Optional[list] = None

    # -- scaling --
    scaling_strategy: str = "baseline"
    bon_n_samples: int = 3
    bon_actor_temperature: float = 0.7
    bon_verifier_model: str = "qwen3-vl"
    bon_verifier_backend: str = "local"  # "local" or "openai_compatible"

    # -- PTS agent settings --
    pts_planner_n: int = 3
    pts_planner_m: int = 1
    pts_planner_temperature: float = 0.7
    pts_translator_n: int = 3
    pts_translator_temperature: float = 0.7
    pts_summarizer_n: int = 3
    pts_summarizer_temperature: float = 0.7
    pts_verifier_model: str = "qwen3-vl"
    pts_verifier_backend: str = "local"

    # -- OpenAI-compatible verifier endpoint (loaded from .env) --
    qwen_api_key: str = ""
    qwen_base_url: str = ""

    @classmethod
    def load(cls, path: str = "config.yaml") -> "AppConfig":
        """Load full config from YAML, applying OS-specific expansions."""
        import os
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        mdl = data.get("model", {})
        bench = data.get("benchmark", {})
        scl = data.get("scaling", {})
        bon = scl.get("best_of_n_weighted", {})
        pts = scl.get("pts_agent", {})

        return cls(
            # model
            model_name=mdl.get("name", cls.model_name),
            model_temperature=mdl.get("temperature", cls.model_temperature),
            model_max_tokens=mdl.get("max_tokens", cls.model_max_tokens),
            model_max_retry=mdl.get("max_retry", cls.model_max_retry),
            # benchmark
            adb_path=os.path.expandvars(
                bench.get("adb_path", cls.adb_path)
            ),
            console_port=bench.get("console_port", cls.console_port),
            perform_emulator_setup=bench.get(
                "perform_emulator_setup", cls.perform_emulator_setup
            ),
            n_task_combinations=bench.get(
                "n_task_combinations", cls.n_task_combinations
            ),
            task_random_seed=bench.get("task_random_seed", cls.task_random_seed),
            output_path=os.path.expanduser(
                bench.get("output_path", cls.output_path)
            ),
            checkpoint_dir=bench.get("checkpoint_dir", cls.checkpoint_dir),
            tasks=bench.get("tasks", cls.tasks),
            # scaling
            scaling_strategy=scl.get("strategy", cls.scaling_strategy),
            bon_n_samples=bon.get("n_samples", cls.bon_n_samples),
            bon_actor_temperature=bon.get(
                "actor_temperature", cls.bon_actor_temperature
            ),
            bon_verifier_model=bon.get(
                "verifier_model", cls.bon_verifier_model
            ),
            bon_verifier_backend=bon.get(
                "verifier_backend", cls.bon_verifier_backend
            ),
            # PTS agent
            pts_planner_n=pts.get("planner_n", cls.pts_planner_n),
            pts_planner_m=pts.get("planner_m", cls.pts_planner_m),
            pts_planner_temperature=pts.get(
                "planner_temperature", cls.pts_planner_temperature
            ),
            pts_translator_n=pts.get("translator_n", cls.pts_translator_n),
            pts_translator_temperature=pts.get(
                "translator_temperature", cls.pts_translator_temperature
            ),
            pts_summarizer_n=pts.get("summarizer_n", cls.pts_summarizer_n),
            pts_summarizer_temperature=pts.get(
                "summarizer_temperature", cls.pts_summarizer_temperature
            ),
            pts_verifier_model=pts.get(
                "verifier_model", cls.pts_verifier_model
            ),
            pts_verifier_backend=pts.get(
                "verifier_backend", cls.pts_verifier_backend
            ),
            # OpenAI-compatible endpoint secrets from environment / .env
            qwen_api_key=os.environ.get("QWEN_API_KEY", cls.qwen_api_key),
            qwen_base_url=os.environ.get("QWEN_BASE_URL", cls.qwen_base_url),
        )

# =============================================================================
# Client
# =============================================================================

def array_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 encoded JPEG string."""
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


class LLMClient:
    """Client for the remote LLM server. Supports text and multimodal requests."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.server_url.rstrip("/")
    
    async def health(self) -> dict:
        """Check server health."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/health", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
    
    async def list_models(self) -> List[str]:
        """List available models."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/v1/models", timeout=10.0)
            resp.raise_for_status()
            return resp.json()["models"]
    
    async def chat(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        images: Optional[List[np.ndarray]] = None
    ) -> str:
        """
        Send chat request to server.
        
        Args:
            model: Model name (e.g., "gelab-zero-4b-preview")
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            images: Optional list of images as numpy arrays
            
        Returns:
            Generated response text
        """
        # Build message content with images if provided
        if images:
            # Convert images to base64 and attach to the last user message
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{array_to_base64(img)}"}
                }
                for img in images
            ]
            # Find last user message and convert to multimodal format
            for msg in reversed(messages):
                if msg["role"] == "user":
                    text_content = msg["content"]
                    msg["content"] = [
                        {"type": "text", "text": text_content},
                        *image_contents
                    ]
                    break
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat",
                json=payload,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Log truncation warning if response was cut off by length limit
            done_reason = data.get("done_reason", "")
            if done_reason == "length":
                eval_count = data.get("eval_count", 0)
                print(f"[LLMClient] WARNING: Response truncated by length limit! eval_count={eval_count}")
            
            return data["response"]

# =============================================================================
# Convenience Functions
# =============================================================================

def get_client(config_path: str = "config.yaml") -> LLMClient:
    """Create client from config file."""
    config = Config.load(config_path)
    return LLMClient(config)


# =============================================================================
# OpenAI-compatible verifier client  (alternative Qwen backend)
# =============================================================================

class OpenAICompatibleVerifierClient:
    """Verifier client that uses the OpenAI Python SDK to call any
    OpenAI-compatible endpoint (e.g. a self-hosted Qwen VL server).

    Credentials are read from environment variables (typically loaded
    from ``.env`` at the top of this module):

    * ``QWEN_API_KEY``  → ``api_key``
    * ``QWEN_BASE_URL`` → ``base_url``  (without trailing ``/v1/``)

    The class exposes the same ``async chat(...)`` signature as
    ``LLMClient`` so it can be used as a drop-in replacement in
    ``BestOfNWeightedStrategy``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ):
        import openai  # lazy import — only needed for this backend

        resolved_key = api_key or os.environ.get("QWEN_API_KEY", "")
        resolved_url = base_url or os.environ.get("QWEN_BASE_URL", "")
        if not resolved_key:
            raise ValueError(
                "QWEN_API_KEY is not set.  Please add it to .env or pass it "
                "explicitly."
            )
        if not resolved_url:
            raise ValueError(
                "QWEN_BASE_URL is not set.  Please add it to .env or pass it "
                "explicitly."
            )

        # Ensure the base_url ends with /v1/ (or /v1)
        if not resolved_url.rstrip("/").endswith("/v1"):
            resolved_url = resolved_url.rstrip("/") + "/v1/"
        if not resolved_url.endswith("/"):
            resolved_url += "/"

        self._client = openai.OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )

    # --- async chat that mirrors LLMClient.chat ---

    async def chat(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        images: Optional[List[np.ndarray]] = None,
    ) -> str:
        """Send a chat completion request via the OpenAI SDK.

        ``messages`` should already be in OpenAI multimodal format
        (list of content parts with ``type: image_url`` etc.) when
        called from the verifier.  If plain-text ``images`` are also
        supplied they are appended to the last user message, matching
        the behaviour of ``LLMClient.chat``.
        """
        import asyncio

        # If bare images are passed separately, fold them into the messages
        if images:
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{array_to_base64(img)}"
                    },
                }
                for img in images
            ]
            for msg in reversed(messages):
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        msg["content"] = [
                            {"type": "text", "text": content},
                            *image_contents,
                        ]
                    elif isinstance(content, list):
                        msg["content"] = content + image_contents
                    break

        # The openai SDK is synchronous by default; run in an executor so
        # we don't block the event loop used for parallel candidate scoring.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return response.choices[0].message.content


async def quick_chat(prompt: str, model: str = "qwen3-vl") -> str:
    """Quick one-off chat request."""
    client = get_client()
    return await client.chat(model, [{"role": "user", "content": prompt}])
