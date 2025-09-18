"""FastAPI router for coding agent endpoints."""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse
import json
import asyncio
import uuid

from agent.schemas import (
    ChatRequest,
    ChatStreamResponse,
    ChatResponse,
    HealthStatus,
    ConversationSummary,
    ModelInfo,
    TaskTypeInfo,
    ModelConfigRequest,
    ModelStatusResponse,
    FileUploadResponse,
    TaskDetectionResult,
)
from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools import get_available_tools
from agent.file_processor import file_processor
from agent.task_detector import task_detector
from llm_server.inference import VLLMEngine
from llm_server.config import LLMConfig
from auth.middleware import get_current_user, require_network_access
from auth.models import UserResponse

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


async def get_memory(project_id: str, user_id: Optional[int] = None) -> ConversationMemory:
    """Get or create memory for a project and user."""
    # Create user-scoped memory key
    memory_key = f"{project_id}_{user_id}" if user_id else project_id

    if memory_key not in _memory_store:
        memory = ConversationMemory(
            max_tokens=2048,
            persist_path=f"data/conversations/{memory_key}.db",
            project_id=memory_key,
        )
        await memory.initialize()
        _memory_store[memory_key] = memory

    return _memory_store[memory_key]


@router.post("/chat", response_model=ChatStreamResponse)
async def chat_stream(
    request: ChatRequest,
    agent = Depends(get_agent),
    current_user: UserResponse = Depends(get_current_user),
    client_ip: str = Depends(require_network_access),
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

        # Get or create user-scoped memory for the project
        memory = await get_memory(request.project_id or "default", current_user.id)

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
    current_user: UserResponse = Depends(get_current_user),
    client_ip: str = Depends(require_network_access),
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

        # Get or create user-scoped memory for the project
        memory = await get_memory(request.project_id or "default", current_user.id)
        agent.memory = memory

        # Add user message with file attachments to memory
        await memory.add_user_message(request.message, request.attached_files)

        # Auto-detect task type if not specified
        final_task_type = request.task_type
        if not final_task_type:
            task_detection = task_detector.detect_task(
                message=request.message,
                attached_files=request.attached_files
            )
            final_task_type = task_detection.detected_task
            logger.info(f"Auto-detected task: {final_task_type} (confidence: {task_detection.confidence:.2f})")

        # Collect full response with model preferences
        response_chunks = []
        async for chunk in agent.chat(
            request.message,
            preferred_model=request.preferred_model,
            task_type=final_task_type
        ):
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
async def list_conversations(
    current_user: UserResponse = Depends(get_current_user)
) -> list[ConversationSummary]:
    """List all active conversations.

    Returns:
        List of conversation summaries
    """
    summaries = []

    # Filter conversations for current user only
    user_prefix = f"_{current_user.id}"
    for project_id, memory in _memory_store.items():
        if memory.messages and (project_id.endswith(user_prefix) or not "_" in project_id):
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
async def delete_conversation(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete a conversation and its history.

    Args:
        project_id: Project identifier

    Returns:
        Deletion confirmation
    """
    try:
        # Create user-scoped memory key
        memory_key = f"{project_id}_{current_user.id}"

        # Try both the user-scoped key and the original project_id
        deleted = False
        for key in [memory_key, project_id]:
            if key in _memory_store:
                memory = _memory_store[key]
                await memory.clear()
                await memory.close()
                del _memory_store[key]
                deleted = True
                break

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"status": "deleted", "project_id": project_id}

    except Exception as e:
        logger.error(f"Error deleting conversation {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_tools(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """List all available tools and their descriptions.

    Returns:
        Dictionary of tools with their metadata
    """
    tools = get_available_tools()

    return {
        "tools": {
            name: {
                "name": name,
                "description": getattr(tool, 'description', f'{name} tool'),
            }
            for name, tool in tools.items()
        },
        "count": len(tools),
    }


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status(
    llm_engine = Depends(get_llm_engine),
    current_user: UserResponse = Depends(get_current_user)
) -> ModelStatusResponse:
    """Get current model configuration status.

    Returns:
        Current model status and configurations
    """
    try:
        # Check if it's a multi-engine
        if hasattr(llm_engine, 'get_engine_status'):
            # MultiEngine case
            engine_status = llm_engine.get_engine_status()

            available_models = []
            task_configurations = []

            # Define task descriptions
            task_descriptions = {
                "code_generation": "Generate new code, functions, and classes",
                "code_review": "Analyze and review existing code",
                "documentation": "Write documentation and explanations",
                "debugging": "Help debug errors and fix issues",
                "general": "General-purpose coding assistance"
            }

            for task, info in engine_status.items():
                # Add model to available models list
                available_models.append(ModelInfo(
                    name=info["model_path"],
                    engine_type="multi",
                    description=info["description"],
                    available=info["loaded"],
                    task_types=[task]
                ))

                # Add task configuration
                task_configurations.append(TaskTypeInfo(
                    task_type=task,
                    description=task_descriptions.get(task, f"Task type: {task}"),
                    current_model=info["model_path"],
                    available_models=[info["model_path"]]
                ))

            return ModelStatusResponse(
                engine_type="multi",
                available_models=available_models,
                task_configurations=task_configurations,
                default_model=available_models[0].name if available_models else "none"
            )
        else:
            # Single engine case
            engine_name = getattr(llm_engine, 'model_name', 'unknown')
            engine_type = type(llm_engine).__name__.lower().replace('engine', '')

            model = ModelInfo(
                name=engine_name,
                engine_type=engine_type,
                description=f"Single {engine_type} model",
                available=True,
                task_types=["general"]
            )

            task_config = TaskTypeInfo(
                task_type="general",
                description="General-purpose assistance",
                current_model=engine_name,
                available_models=[engine_name]
            )

            return ModelStatusResponse(
                engine_type=engine_type,
                available_models=[model],
                task_configurations=[task_config],
                default_model=engine_name
            )

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/available")
async def list_available_models(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """List all models available in the system.

    Returns:
        Dictionary of available models and their info
    """
    try:
        # This would be enhanced to scan Ollama and other engines
        # For now, return basic info
        ollama_models = []

        # Try to get Ollama models if available
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        ollama_models.append({
                            "name": model_name,
                            "engine_type": "ollama",
                            "available": True
                        })
        except:
            pass  # Ollama not available

        return {
            "ollama_models": ollama_models,
            "total_available": len(ollama_models)
        }

    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/configure")
async def configure_model(
    request: ModelConfigRequest,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, str]:
    """Configure model assignment for a task type.

    Args:
        request: Model configuration request

    Returns:
        Success message
    """
    try:
        # For now, this is a placeholder - real implementation would update
        # the MultiEngine configuration and persist it
        logger.info(f"Model configuration request: {request.task_type} -> {request.model_name}")

        return {
            "status": "success",
            "message": f"Task '{request.task_type}' configured to use model '{request.model_name}'"
        }

    except Exception as e:
        logger.error(f"Error configuring model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user),
    client_ip: str = Depends(require_network_access),
) -> FileUploadResponse:
    """Upload a file for processing and analysis.

    Args:
        file: Uploaded file
        current_user: Current authenticated user

    Returns:
        File upload response with processing status
    """
    try:
        # Read file content
        content = await file.read()

        # Validate file
        validation = file_processor.validate_file(file.filename or "unknown", content)

        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {', '.join(validation['errors'])}"
            )

        # Save file
        file_id = file_processor.save_file(file.filename or "unknown", content)

        # Extract content
        processed_content = file_processor.extract_content(file_id, file.filename or "unknown")
        processed = processed_content is not None

        logger.info(f"File uploaded: {file.filename} -> {file_id} (processed: {processed})")

        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename or "unknown",
            content_type=validation["content_type"] or "application/octet-stream",
            file_size=validation["file_size"],
            processed=processed,
            message="File uploaded and processed successfully" if processed else "File uploaded but content extraction failed"
        )

    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}")
async def get_file_info(
    file_id: str,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get information about an uploaded file.

    Args:
        file_id: Unique file identifier

    Returns:
        File information
    """
    try:
        file_info = file_processor.get_file_info(file_id)

        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")

        return file_info

    except Exception as e:
        logger.error(f"Error getting file info {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete an uploaded file.

    Args:
        file_id: Unique file identifier

    Returns:
        Deletion confirmation
    """
    try:
        success = file_processor.delete_file(file_id)

        if not success:
            raise HTTPException(status_code=404, detail="File not found")

        return {"status": "deleted", "file_id": file_id}

    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/detect", response_model=TaskDetectionResult)
async def detect_task_type(
    message: str,
    attached_files: Optional[List[str]] = None,
    current_user: UserResponse = Depends(get_current_user)
) -> TaskDetectionResult:
    """Detect the most likely task type for a message.

    Args:
        message: User message to analyze
        attached_files: List of attached filenames

    Returns:
        Task detection result with confidence and suggestions
    """
    try:
        result = task_detector.detect_task(
            message=message,
            attached_files=attached_files
        )

        logger.info(f"Task detected: {result.detected_task} (confidence: {result.confidence:.2f})")
        return result

    except Exception as e:
        logger.error(f"Task detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/types")
async def get_task_types(
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, str]:
    """Get all available task types and their descriptions.

    Returns:
        Dictionary of task types and descriptions
    """
    try:
        return task_detector.get_task_descriptions()
    except Exception as e:
        logger.error(f"Error getting task types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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