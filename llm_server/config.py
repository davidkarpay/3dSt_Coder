"""Configuration for LLM inference server."""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class LLMConfig(BaseSettings):
    """Configuration settings for vLLM engine.

    All settings can be overridden via environment variables with LLM_ prefix.
    Example: LLM_MODEL_PATH=/path/to/model
    """

    model_path: str = Field(
        default="/models/deepseek-moe-coder",
        description="Path to the model weights directory",
    )
    gpu_id: int = Field(
        default=0,
        description="GPU device ID to use for inference",
    )
    max_context: int = Field(
        default=2048,
        description="Maximum context length in tokens",
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum tokens to generate per request",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind the server to",
    )
    port: int = Field(
        default=8001,
        description="Port for the LLM inference server",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    class Config:
        """Pydantic config."""

        env_prefix = "LLM_"
        env_file = ".env"
        env_file_encoding = "utf-8"