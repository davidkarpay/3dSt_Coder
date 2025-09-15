"""Pydantic schemas for agent interactions."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Possible message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Represents a single message in a conversation."""

    role: MessageRole = Field(description="The role of the message sender")
    content: str = Field(description="The message content")
    timestamp: Optional[datetime] = Field(
        default=None, description="When the message was created"
    )
    tool: Optional[str] = Field(default=None, description="Tool name if role is 'tool'")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )


class ToolCall(BaseModel):
    """Represents a tool invocation request."""

    name: str = Field(description="Name of the tool to invoke")
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    tool_name: str = Field(description="Name of the executed tool")
    result: str = Field(description="Tool execution result")
    success: bool = Field(description="Whether the tool executed successfully")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(
        default=None, description="Execution time in seconds"
    )


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(
        description="User message to send to the agent",
        min_length=1,
        max_length=10000,
    )
    stream: bool = Field(
        default=True, description="Whether to stream the response"
    )
    project_id: Optional[str] = Field(
        default=None, description="Project identifier for conversation scoping"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Override maximum tokens to generate",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override sampling temperature",
    )


class ChatStreamResponse(BaseModel):
    """Streaming response schema for chat."""

    delta: str = Field(description="Incremental response chunk")
    finish_reason: Optional[str] = Field(
        default=None, description="Reason for completion (length, stop, tool_call)"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Any tool calls made during generation"
    )


class ChatResponse(BaseModel):
    """Complete response schema for non-streaming chat."""

    response: str = Field(description="Complete agent response")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tool calls made during generation"
    )
    tokens_used: int = Field(description="Total tokens used in generation")
    conversation_id: str = Field(description="Conversation session identifier")


class AgentResponse(BaseModel):
    """Agent response with metadata."""

    content: str = Field(description="Response content")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tool calls made"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Agent's reasoning process"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the response",
    )


class ConversationSummary(BaseModel):
    """Summary of a conversation session."""

    conversation_id: str = Field(description="Conversation identifier")
    project_id: str = Field(description="Project identifier")
    message_count: int = Field(description="Total number of messages")
    start_time: datetime = Field(description="When conversation started")
    last_activity: datetime = Field(description="Last message timestamp")
    tool_usage: Dict[str, int] = Field(
        description="Count of tool usage by tool name"
    )
    total_tokens: int = Field(description="Estimated total tokens used")


class HealthStatus(BaseModel):
    """Agent health status."""

    status: str = Field(description="Overall status (healthy, degraded, unhealthy)")
    llm_available: bool = Field(description="Whether LLM engine is available")
    tools_available: List[str] = Field(description="List of available tools")
    memory_usage: Dict[str, Any] = Field(description="Memory usage statistics")
    uptime_seconds: float = Field(description="Agent uptime in seconds")