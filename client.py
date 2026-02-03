# =============================================================================
# test_server.py
# Created:  [Lc-v0.0.1] 2026-02-03
# Updated:  [Lc-v0.0.1] 2026-02-03 Implements server and client
# =============================================================================
"""
Client for remote LLM inference server.
"""

import yaml
import httpx
import asyncio
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

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

class LLMClient:
    """Simple client for the remote LLM server."""
    
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
        max_tokens: int = 2048
    ) -> str:
        """
        Send chat request to server.
        
        Args:
            model: Model name ("gelab-zero-4b-preview" or "qwen3-vl")
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Generated response text
        """
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
            return resp.json()["response"]

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
