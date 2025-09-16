"""OpenAI API engine for cloud-based inference."""

import asyncio
import httpx
import json
import os
from typing import AsyncIterator, Optional, List
import logging

from .config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIEngine:
    """OpenAI API engine wrapper."""

    def __init__(self, cfg: LLMConfig):
        """Initialize with configuration."""
        self.cfg = cfg
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.model_name = cfg.model_path or "gpt-4"

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate streaming response from OpenAI API."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": temperature or self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "top_p": self.cfg.top_p,
            "stop": stop,
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip() and line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix

                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            yield f"Error: {str(e)}"

    async def shutdown(self) -> None:
        """Cleanup (nothing needed for HTTP client)."""
        pass