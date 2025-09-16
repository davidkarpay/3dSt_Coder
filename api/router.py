"""FastAPI router for coding agent endpoints."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse
import json
import asyncio

from agent.schemas import (
    ChatRequest,
    ChatStreamResponse,
    ChatResponse,
    HealthStatus,
    ConversationSummary,
)
from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools import get_available_tools
from llm_server.inference import VLLMEngine
from llm_server.config import LLMConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["3dst-coder"])

# Global instances (will be initialized in main.py)
_agent_instance = None
_llm_engine = None
_memory_store: Dict[str, ConversationMemory] = {}


def get_agent():
    """Dependency to get the global agent instance."""
    if _agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent_instance


def get_llm_engine():
    """Dependency to get the global LLM engine."""
    if _llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not initialized")
    return _llm_engine


async def get_memory(project_id: str) -> ConversationMemory:
    """Get or create memory for a project."""
    if project_id not in _memory_store:
        memory = ConversationMemory(
            max_tokens=2048,
            persist_path=f"data/conversations/{project_id}.db",
            project_id=project_id,
        )
        await memory.initialize()
        _memory_store[project_id] = memory

    return _memory_store[project_id]


@router.post("/chat", response_model=ChatStreamResponse)
async def chat_stream(
    request: ChatRequest,
    agent = Depends(get_agent),
) -> EventSourceResponse:
    """Stream a chat response from the coding agent.

    Args:
        request: Chat request with message and options
        agent: Injected agent instance

    Returns:
        Server-sent events stream with response chunks
    """
    try:
        logger.info(f"Chat request: {request.message[:100]}...")

        # Get or create memory for the project
        memory = await get_memory(request.project_id or "default")

        # Update agent's memory for this session
        agent.memory = memory

        async def event_generator():
            """Generate server-sent events for streaming response."""
            try:
                tool_calls = []

                async for chunk in agent.chat(request.message):
                    # Create response data
                    data = {
                        "delta": chunk,
                        "tool_calls": tool_calls,
                        "finish_reason": None,
                    }

                    # Send as SSE event
                    yield {
                        "event": "message",
                        "data": json.dumps(data),
                    }

                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)

                # Send completion event
                completion_data = {
                    "delta": "",
                    "tool_calls": tool_calls,
                    "finish_reason": "stop",
                }
                yield {
                    "event": "complete",
                    "data": json.dumps(completion_data),
                }

            except Exception as e:
                logger.error(f"Error in chat stream: {e}")
                error_data = {
                    "error": str(e),
                    "finish_reason": "error",
                }
                yield {
                    "event": "error",
                    "data": json.dumps(error_data),
                }

        return EventSourceResponse(event_generator())

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/complete", response_model=ChatResponse)
async def chat_complete(
    request: ChatRequest,
    agent = Depends(get_agent),
) -> ChatResponse:
    """Get a complete chat response (non-streaming).

    Args:
        request: Chat request with message and options
        agent: Injected agent instance

    Returns:
        Complete chat response
    """
    try:
        logger.info(f"Complete chat request: {request.message[:100]}...")

        # Get or create memory for the project
        memory = await get_memory(request.project_id or "default")
        agent.memory = memory

        # Collect full response
        response_chunks = []
        async for chunk in agent.chat(request.message):
            response_chunks.append(chunk)

        full_response = "".join(response_chunks)

        return ChatResponse(
            response=full_response,
            tool_calls=[],  # TODO: Extract tool calls from response
            tokens_used=len(full_response) // 4,  # Rough estimation
            conversation_id=request.project_id or "default",
        )

    except Exception as e:
        logger.error(f"Complete chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthStatus)
async def health_check(
    llm_engine = Depends(get_llm_engine),
) -> HealthStatus:
    """Check the health status of the coding agent.

    Returns:
        Health status including LLM and tool availability
    """
    try:
        # Check LLM availability
        llm_available = llm_engine is not None

        # Get available tools
        available_tools = list(get_available_tools().keys())

        # Basic memory usage (simplified)
        memory_usage = {
            "active_conversations": len(_memory_store),
            "total_messages": sum(
                len(memory.messages) for memory in _memory_store.values()
            ),
        }

        return HealthStatus(
            status="healthy" if llm_available else "degraded",
            llm_available=llm_available,
            tools_available=available_tools,
            memory_usage=memory_usage,
            uptime_seconds=0.0,  # TODO: Track actual uptime
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthStatus(
            status="unhealthy",
            llm_available=False,
            tools_available=[],
            memory_usage={},
            uptime_seconds=0.0,
        )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations() -> list[ConversationSummary]:
    """List all active conversations.

    Returns:
        List of conversation summaries
    """
    summaries = []

    for project_id, memory in _memory_store.items():
        if memory.messages:
            first_msg = memory.messages[0]
            last_msg = memory.messages[-1]

            # Count tool usage
            tool_usage = {}
            for msg in memory.messages:
                if msg.role == "tool" and msg.tool:
                    tool_usage[msg.tool] = tool_usage.get(msg.tool, 0) + 1

            summaries.append(
                ConversationSummary(
                    conversation_id=project_id,
                    project_id=project_id,
                    message_count=len(memory.messages),
                    start_time=first_msg.timestamp,
                    last_activity=last_msg.timestamp,
                    tool_usage=tool_usage,
                    total_tokens=sum(len(msg.content) for msg in memory.messages) // 4,
                )
            )

    return summaries


@router.delete("/conversations/{project_id}")
async def delete_conversation(project_id: str) -> Dict[str, str]:
    """Delete a conversation and its history.

    Args:
        project_id: Project identifier

    Returns:
        Deletion confirmation
    """
    try:
        if project_id in _memory_store:
            memory = _memory_store[project_id]
            await memory.clear()
            await memory.close()
            del _memory_store[project_id]

        return {"status": "deleted", "project_id": project_id}

    except Exception as e:
        logger.error(f"Error deleting conversation {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_tools() -> Dict[str, Any]:
    """List all available tools and their descriptions.

    Returns:
        Dictionary of tools with their metadata
    """
    tools = get_available_tools()

    return {
        "tools": {
            name: {
                "name": tool.name,
                "description": tool.description,
            }
            for name, tool in tools.items()
        },
        "count": len(tools),
    }


# Initialization function to be called from main.py
async def initialize_router(
    llm_engine,
    agent,
) -> None:
    """Initialize the router with global instances.

    Args:
        llm_engine: Initialized LLM engine
        agent: Initialized coding agent
    """
    global _agent_instance, _llm_engine

    _agent_instance = agent
    _llm_engine = llm_engine

    logger.info("API router initialized with agent and LLM engine")