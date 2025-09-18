#!/usr/bin/env python3
"""
Startup script for 3dSt Platform with Ollama backend.

Prerequisites:
1. Ollama installed and running (ollama serve)
2. A model pulled (e.g., ollama pull codellama:7b)
"""

import os
import sys
import asyncio
import subprocess
import time
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("[OK] Ollama server is running")
            return True
    except:
        pass

    print("[ERROR] Ollama server not found. Please start it with: ollama serve")
    return False

def check_ollama_model(model_name):
    """Check if the specified model is available."""
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=10
        )
        if response.status_code == 200:
            print(f"[OK] Model '{model_name}' is available")
            return True
    except:
        pass

    print(f"[ERROR] Model '{model_name}' not found. Pull it with: ollama pull {model_name}")
    return False

def setup_environment():
    """Set up environment variables for Ollama."""
    # Default model (can be overridden)
    model_name = os.getenv("LLM_MODEL_PATH", "codellama:7b")

    # Check if model exists
    if not check_ollama_model(model_name):
        # Try some common coding models
        common_models = ["codellama:7b", "deepseek-coder:6.7b", "codegemma:7b", "llama3:8b"]

        print(f"\nTrying to find an available model...")
        for model in common_models:
            if check_ollama_model(model):
                model_name = model
                break
        else:
            print("\nNo suitable models found. Please pull a model:")
            print("  ollama pull codellama:7b")
            print("  ollama pull deepseek-coder:6.7b")
            print("  ollama pull codegemma:7b")
            return False

    # Set environment variables
    os.environ["LLM_ENGINE_TYPE"] = "ollama"
    os.environ["LLM_MODEL_PATH"] = model_name
    os.environ["LLM_HOST"] = "127.0.0.1"
    os.environ["LLM_PORT"] = "8000"

    print(f"[START] Starting 3dSt Platform with model: {model_name}")
    return True

def main():
    """Main startup function."""
    print("3dSt Platform + Ollama Startup")
    print("=" * 40)

    # Check Ollama
    if not check_ollama():
        print("\n[INFO] To start Ollama:")
        print("   ollama serve")
        sys.exit(1)

    # Setup environment
    if not setup_environment():
        sys.exit(1)

    # Start 3dSt_Coder
    print("\n[START] Starting 3dSt Platform API server...")
    print("[WEB] Web interface will be available at: http://localhost:8000")
    print("[API] API docs available at: http://localhost:8000/docs")
    print("\n[EXAMPLES] Example prompts to try:")
    print("   • 'Check the git status of this repository'")
    print("   • 'Read the README.md file and summarize it'")
    print("   • 'Write a Python function to calculate fibonacci numbers'")
    print("   • 'Run the tests in this project'")
    print("\nPress Ctrl+C to stop")
    print("=" * 40)

    try:
        # Import and run the main app
        from api.main import app
        import uvicorn

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down 3dSt Platform...")
    except Exception as e:
        print(f"\n[ERROR] Error starting 3dSt Platform: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()