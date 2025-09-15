"""vLLM inference engine wrapper for async token streaming."""

from typing import AsyncIterator, List, Optional
import logging
import asyncio

from .config import LLMConfig

# Conditional import - vLLM may not be available in test environment
try:
    from vllm import AsyncLLMEngine, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Mock classes for when vLLM is not installed
    class AsyncLLMEngine:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("vLLM not installed. Install with: pip install vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

logger = logging.getLogger(__name__)


class VLLMEngine:
    """Thin wrapper around vLLM that handles streaming and MoE-specific settings."""

    def __init__(self, cfg: LLMConfig):
        """Initialize the engine with configuration.

        Args:
            cfg: LLMConfig instance with model settings
        """
        self.cfg = cfg
        self._engine: Optional[AsyncLLMEngine] = None
        self._lock = asyncio.Lock()  # For thread-safe lazy initialization

    async def _initialize_engine(self) -> None:
        """Lazily initialize the vLLM engine on first use."""
        if self._engine is not None:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self._engine is not None:
                return

            logger.info(f"Initializing vLLM engine with model: {self.cfg.model_path}")

            try:
                self._engine = await AsyncLLMEngine.from_pretrained(
                    self.cfg.model_path,
                    tensor_parallel_size=1,  # Single GPU for now
                    gpu_memory_utilization=0.9,
                    max_model_len=self.cfg.max_context,
                    enforce_eager=True,  # Required for MoE models
                    # Additional MoE-specific flags can be added here
                )
                logger.info("vLLM engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM engine: {e}")
                raise

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate tokens asynchronously with streaming.

        Args:
            prompt: Input text prompt
            stop: Optional list of stop sequences
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            Generated tokens as they become available
        """
        # Initialize engine if needed
        if self._engine is None:
            await self._initialize_engine()

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature or self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_tokens or self.cfg.max_tokens,
            stop=stop,
        )

        try:
            # Generate tokens
            logger.debug(f"Starting generation for prompt: {prompt[:100]}...")

            # For testing/development without vLLM
            if not VLLM_AVAILABLE or self._engine is None:
                logger.warning("vLLM not available, using mock generation")
                # Simple mock that yields prompt back in chunks
                words = prompt.split()
                for word in words:
                    yield word + " " if word != words[-1] else word
                return

            # Real vLLM generation
            request_id = f"req_{id(prompt)}"  # Unique request ID

            async for output in self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                # vLLM returns RequestOutput objects
                # Extract the generated text delta
                if output.outputs:
                    for completion_output in output.outputs:
                        # Yield only the new tokens (delta)
                        if completion_output.text:
                            yield completion_output.text

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the vLLM engine and free resources."""
        if self._engine is not None:
            logger.info("Shutting down vLLM engine")
            # vLLM engine cleanup if needed
            self._engine = None