"""Pydantic schemas for LLM server API."""

from pydantic import BaseModel, Field
from typing import Optional, List


class GenerateRequest(BaseModel):
    """Request schema for text generation."""

    prompt: str = Field(
        ...,
        description="Input prompt for text generation",
        min_length=1,
        max_length=10000,
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="List of sequences where generation should stop",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override sampling temperature",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Override maximum tokens to generate",
    )

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "prompt": "def fibonacci(n):",
                "stop": ["\n\n", "def "],
                "temperature": 0.8,
                "max_tokens": 256,
            }
        }


class GenerateResponse(BaseModel):
    """Response schema for streaming generation."""

    token: str = Field(
        ...,
        description="Generated token chunk",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for generation completion (length, stop, etc.)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_path: str = Field(description="Path to loaded model")
    gpu_available: bool = Field(description="Whether GPU is available")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(description="HTTP status code")