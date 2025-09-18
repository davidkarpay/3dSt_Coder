"""Ollama engine implementation for local model serving."""

import asyncio
import httpx
import json
from typing import AsyncIterator, Optional, List
import logging

from .config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaEngine:
    """Ollama engine wrapper for local model serving."""

    def __init__(self, cfg: LLMConfig):
        """Initialize with configuration."""
        self.cfg = cfg
        self.base_url = f"http://{cfg.host}:11434"  # Ollama default port
        self.model_name = cfg.model_path.split("/")[-1]  # Extract model name

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate streaming response from Ollama."""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature or self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": max_tokens or self.cfg.max_tokens,
                "stop": stop or [],
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.cfg.request_timeout
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")

            # Provide specific error messages for common issues
            error_str = str(e).lower()
            if "timeout" in error_str or "time" in error_str:
                yield f"Error: Request timed out after {self.cfg.request_timeout}s. The model may be busy or the query too complex. Try a simpler request."
            elif "connection" in error_str:
                yield "Error: Could not connect to Ollama server. Please ensure Ollama is running on port 11434."
            elif "model" in error_str:
                yield f"Error: Model '{self.model_name}' not found. Please check the model name or pull it with 'ollama pull {self.model_name}'."
            else:
                yield f"Error: {str(e)}"

    async def shutdown(self) -> None:
        """Cleanup (nothing needed for Ollama)."""
        pass