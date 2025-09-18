"""Multi-engine system for task-specific model routing."""

import os
from typing import Dict, Any, AsyncIterator, Optional, List
import logging

from .config import LLMConfig
from .engine_factory import EngineFactory

logger = logging.getLogger(__name__)


class MultiEngine:
    """Route different tasks to specialized models."""

    def __init__(self, base_config: LLMConfig):
        """Initialize with base configuration."""
        self.base_config = base_config
        self.engines: Dict[str, Any] = {}
        self._load_task_engines()

    def _load_task_engines(self):
        """Load engines for different task types."""
        # Task-specific model configurations
        task_configs = {
            "code_generation": {
                "engine_type": "vllm",
                "model_path": "/models/deepseek-coder-7b",
                "description": "Specialized for code generation"
            },
            "code_review": {
                "engine_type": "ollama",
                "model_path": "llama3:8b",
                "description": "Good for code analysis and review"
            },
            "documentation": {
                "engine_type": "openai",
                "model_path": "gpt-4",
                "description": "Excellent for writing documentation"
            },
            "debugging": {
                "engine_type": "vllm",
                "model_path": "/models/codellama-instruct-7b",
                "description": "Optimized for debugging assistance"
            },
            "general": {
                "engine_type": "vllm",
                "model_path": "/models/deepseek-moe-coder",
                "description": "General purpose coding assistant"
            }
        }

        for task, config in task_configs.items():
            try:
                # Create task-specific config
                task_config = LLMConfig(
                    model_path=config["model_path"],
                    **self.base_config.model_dump()
                )

                # Set engine type via environment (temporary)
                original_engine_type = os.getenv("LLM_ENGINE_TYPE")
                os.environ["LLM_ENGINE_TYPE"] = config["engine_type"]

                # Create engine
                engine = EngineFactory.create_engine(task_config)
                self.engines[task] = {
                    "engine": engine,
                    "config": task_config,
                    "description": config["description"]
                }

                # Restore original engine type
                if original_engine_type:
                    os.environ["LLM_ENGINE_TYPE"] = original_engine_type

                logger.info(f"Loaded {task} engine: {config['engine_type']} - {config['model_path']}")

            except Exception as e:
                logger.warning(f"Failed to load {task} engine: {e}")

    def detect_task_type(self, prompt: str) -> str:
        """Detect task type from prompt content."""
        prompt_lower = prompt.lower()

        # Simple heuristics - could be enhanced with ML
        if any(word in prompt_lower for word in ["write function", "implement", "create class", "generate code"]):
            return "code_generation"

        elif any(word in prompt_lower for word in ["review", "check code", "analyze", "feedback"]):
            return "code_review"

        elif any(word in prompt_lower for word in ["document", "explain", "describe", "readme"]):
            return "documentation"

        elif any(word in prompt_lower for word in ["debug", "error", "fix", "bug", "exception"]):
            return "debugging"

        else:
            return "general"

    async def generate(
        self,
        prompt: str,
        task_type: Optional[str] = None,
        preferred_model: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate response using appropriate model for the task."""

        # If preferred_model is specified, try to find engine with that model
        if preferred_model:
            for task, engine_info in self.engines.items():
                if preferred_model in engine_info["config"].model_path:
                    task_type = task
                    logger.info(f"Using preferred model '{preferred_model}' from {task} engine")
                    break

        # Auto-detect task if not specified
        if task_type is None:
            task_type = self.detect_task_type(prompt)

        # Fall back to general if task not available
        if task_type not in self.engines:
            logger.warning(f"Task type '{task_type}' not available, using general")
            task_type = "general"

        # Use general as final fallback
        if task_type not in self.engines:
            task_type = list(self.engines.keys())[0] if self.engines else None

        if not task_type or task_type not in self.engines:
            yield "Error: No engines available"
            return

        engine = self.engines[task_type]["engine"]
        model_name = self.engines[task_type]["config"].model_path
        logger.info(f"Using {task_type} engine ({model_name}) for generation")

        try:
            async for token in engine.generate(prompt, stop=stop, **kwargs):
                yield token
        except Exception as e:
            logger.error(f"Generation error with {task_type} engine: {e}")
            yield f"Error: {str(e)}"

    async def shutdown(self):
        """Shutdown all engines."""
        for task, engine_info in self.engines.items():
            try:
                await engine_info["engine"].shutdown()
                logger.info(f"Shutdown {task} engine")
            except Exception as e:
                logger.error(f"Error shutting down {task} engine: {e}")

    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all loaded engines."""
        return {
            task: {
                "model_path": info["config"].model_path,
                "description": info["description"],
                "loaded": True
            }
            for task, info in self.engines.items()
        }