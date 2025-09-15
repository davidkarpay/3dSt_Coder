import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import AsyncIterator

from llm_server.inference import VLLMEngine
from llm_server.config import LLMConfig


class TestVLLMEngine:
    """Test suite for VLLMEngine following TDD principles."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock LLM configuration."""
        cfg = Mock(spec=LLMConfig)
        cfg.model_path = "deepseek-moe-coder"
        cfg.gpu_id = 0
        cfg.max_context = 2048
        cfg.max_tokens = 512
        cfg.temperature = 0.7
        cfg.top_p = 0.95
        return cfg

    @pytest.mark.asyncio
    async def test_generate_streams_tokens(self, mock_config):
        """Test that generate method yields tokens as they become available."""
        engine = VLLMEngine(mock_config)

        # Mock the actual vLLM engine
        mock_vllm = AsyncMock()
        async def mock_generate(*args, **kwargs):
            for token in ["def", " hello", "():", " pass"]:
                yield token

        mock_vllm.generate = mock_generate
        engine._engine = mock_vllm

        tokens = []
        async for t in engine.generate("def hello(): pass"):
            tokens.append(t)

        assert "".join(tokens) == "def hello(): pass"
        assert len(tokens) == 4  # Streamed in 4 chunks

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, mock_config):
        """Test that vLLM engine is lazily initialized on first use."""
        engine = VLLMEngine(mock_config)

        # Engine should not be initialized yet
        assert engine._engine is None

        # Mock the vLLM import and initialization
        with patch('llm_server.inference.AsyncLLMEngine') as mock_vllm_class:
            mock_instance = AsyncMock()
            mock_vllm_class.from_pretrained.return_value = mock_instance

            async def mock_generate(*args, **kwargs):
                yield "test"
            mock_instance.generate = mock_generate

            # First generation should initialize the engine
            async for _ in engine.generate("test"):
                pass

            # Verify initialization happened
            mock_vllm_class.from_pretrained.assert_called_once_with(
                mock_config.model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=mock_config.max_context,
                enforce_eager=True  # For MoE models
            )
            assert engine._engine is not None

    @pytest.mark.asyncio
    async def test_generate_with_stop_sequences(self, mock_config):
        """Test generation with stop sequences."""
        engine = VLLMEngine(mock_config)

        mock_vllm = AsyncMock()
        async def mock_generate(prompt, sampling_params):
            # Verify stop sequences are passed correctly
            assert sampling_params.stop == ["\n\n", "END"]
            yield "Generated"
            yield " text"

        mock_vllm.generate = mock_generate
        engine._engine = mock_vllm

        tokens = []
        async for t in engine.generate("prompt", stop=["\n\n", "END"]):
            tokens.append(t)

        assert "".join(tokens) == "Generated text"

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, mock_config):
        """Test that max_tokens from config is respected."""
        mock_config.max_tokens = 10
        engine = VLLMEngine(mock_config)

        with patch('llm_server.inference.SamplingParams') as mock_sampling:
            mock_vllm = AsyncMock()
            async def mock_generate(*args, **kwargs):
                yield "test"

            mock_vllm.generate = mock_generate
            engine._engine = mock_vllm

            async for _ in engine.generate("test"):
                pass

            # Verify SamplingParams was called with correct max_tokens
            mock_sampling.assert_called_with(
                temperature=mock_config.temperature,
                top_p=mock_config.top_p,
                max_tokens=10,
                stop=None
            )

    @pytest.mark.asyncio
    async def test_handle_generation_error(self, mock_config):
        """Test error handling during generation."""
        engine = VLLMEngine(mock_config)

        mock_vllm = AsyncMock()
        async def mock_generate(*args, **kwargs):
            yield "Start"
            raise RuntimeError("GPU out of memory")

        mock_vllm.generate = mock_generate
        engine._engine = mock_vllm

        tokens = []
        with pytest.raises(RuntimeError) as exc_info:
            async for t in engine.generate("test"):
                tokens.append(t)

        assert "GPU out of memory" in str(exc_info.value)
        assert tokens == ["Start"]  # Partial result before error

    @pytest.mark.asyncio
    async def test_concurrent_generation_requests(self, mock_config):
        """Test that engine handles concurrent generation requests."""
        engine = VLLMEngine(mock_config)

        mock_vllm = AsyncMock()
        call_count = 0

        async def mock_generate(prompt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Different responses for different prompts
            if "first" in prompt:
                for token in ["First", " response"]:
                    yield token
            else:
                for token in ["Second", " response"]:
                    yield token

        mock_vllm.generate = mock_generate
        engine._engine = mock_vllm

        # Run two generations concurrently
        async def gen1():
            tokens = []
            async for t in engine.generate("first prompt"):
                tokens.append(t)
            return "".join(tokens)

        async def gen2():
            tokens = []
            async for t in engine.generate("second prompt"):
                tokens.append(t)
            return "".join(tokens)

        results = await asyncio.gather(gen1(), gen2())

        assert results[0] == "First response"
        assert results[1] == "Second response"
        assert call_count == 2