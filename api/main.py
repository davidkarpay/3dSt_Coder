"""FastAPI application entry point for the coding agent API."""

import logging
import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.router import router, initialize_router
from agent.core import CodingAgent
from agent.memory import ConversationMemory
from agent.tools import get_available_tools
from llm_server.inference import VLLMEngine
from llm_server.config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
llm_engine: VLLMEngine = None
coding_agent: CodingAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    global llm_engine, coding_agent

    # Startup
    logger.info("Starting coding agent API...")

    try:
        # Initialize LLM configuration
        llm_config = LLMConfig()
        logger.info(f"LLM Config: {llm_config.model_path}")

        # Initialize LLM engine
        llm_engine = VLLMEngine(llm_config)
        logger.info("LLM engine initialized")

        # Initialize tools
        tools = get_available_tools()
        logger.info(f"Loaded {len(tools)} tools: {list(tools.keys())}")

        # Initialize default memory (will be replaced per-session)
        default_memory = ConversationMemory(
            max_tokens=llm_config.max_context,
            persist_path="data/conversations/default.db",
            project_id="default",
        )
        await default_memory.initialize()

        # Initialize coding agent
        coding_agent = CodingAgent(
            llm=llm_engine,
            tools=tools,
            memory=default_memory,
        )
        logger.info("Coding agent initialized")

        # Initialize router with dependencies
        await initialize_router(llm_engine, coding_agent)

        logger.info("Coding agent API started successfully")

    except Exception as e:
        logger.error(f"Failed to start coding agent API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down coding agent API...")

    try:
        # Close LLM engine
        if llm_engine:
            await llm_engine.shutdown()

        # Close default memory
        if coding_agent and coding_agent.memory:
            await coding_agent.memory.close()

        logger.info("Coding agent API shut down successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Local LLM Coding Agent API",
    description="A locally-hosted coding agent with ReAct loop and tool integration",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/ping")
async def ping():
    """Simple ping endpoint for basic health checks."""
    return {"status": "pong", "message": "Coding agent API is running"}


if __name__ == "__main__":
    import uvicorn

    # Create data directories
    os.makedirs("data/conversations", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Run the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True,
    )