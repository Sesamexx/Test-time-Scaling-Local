# =============================================================================
# client.py
# Created:  [Lc-v0.0.1] 2026-02-03
# Updated:  [Lc-v0.0.2] 2026-02-04 Add multimodal support for images
# =============================================================================
"""
Client for remote LLM inference server.
Supports both text-only and multimodal (image) requests.
"""

import yaml
import httpx
import asyncio
import base64
import io
from pathlib import Path
from typing import List, Optional, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    server_url: str
    timeout: float = 300.0
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            server_url=data["server"]["url"],
            timeout=data["server"].get("timeout", 300.0)
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

async def quick_chat(prompt: str, model: str = "qwen3-vl") -> str:
    """Quick one-off chat request."""
    client = get_client()
    return await client.chat(model, [{"role": "user", "content": prompt}])
