"""Factory for creating different LLM engines based on configuration."""

import os
from typing import Any
from .config import LLMConfig
from .inference import VLLMEngine


class EngineFactory:
    """Factory for creating LLM engines based on configuration."""

    @staticmethod
    def create_engine(cfg: LLMConfig) -> Any:
        """Create appropriate engine based on configuration.

        Args:
            cfg: LLM configuration

        Returns:
            Configured LLM engine instance
        """
        engine_type = os.getenv("LLM_ENGINE_TYPE", "vllm").lower()

        if engine_type == "vllm":
            return VLLMEngine(cfg)

        elif engine_type == "ollama":
            from .ollama_engine import OllamaEngine
            return OllamaEngine(cfg)

        elif engine_type == "openai":
            from .openai_engine import OpenAIEngine
            return OpenAIEngine(cfg)

        elif engine_type == "transformers":
            from .transformers_engine import TransformersEngine
            return TransformersEngine(cfg)

        elif engine_type == "llama_cpp":
            from .llama_cpp_engine import LlamaCppEngine
            return LlamaCppEngine(cfg)

        elif engine_type == "multi":
            from .multi_engine import MultiEngine
            return MultiEngine(cfg)

        else:
            raise ValueError(f"Unknown engine type: {engine_type}")


# Usage examples:
"""
# Use vLLM (default)
export LLM_ENGINE_TYPE=vllm
export LLM_MODEL_PATH=/models/deepseek-moe-coder

# Use Ollama
export LLM_ENGINE_TYPE=ollama
export LLM_MODEL_PATH=codellama:7b

# Use OpenAI API
export LLM_ENGINE_TYPE=openai
export LLM_MODEL_PATH=gpt-4
export OPENAI_API_KEY=sk-...

# Use HuggingFace Transformers
export LLM_ENGINE_TYPE=transformers
export LLM_MODEL_PATH=microsoft/DialoGPT-medium

# Use llama.cpp
export LLM_ENGINE_TYPE=llama_cpp
export LLM_MODEL_PATH=/models/llama-2-7b.gguf

# Use multi-engine (task-based routing)
export LLM_ENGINE_TYPE=multi
"""