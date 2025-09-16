#!/usr/bin/env python3
"""Start 3dSt_Coder with your Saul Instruct model."""

import os
import sys

# Set environment for your specific model
os.environ["LLM_ENGINE_TYPE"] = "ollama"
os.environ["LLM_MODEL_PATH"] = "adrienbrault/saul-instruct-v1:q4_k_m"
os.environ["LLM_HOST"] = "127.0.0.1"
os.environ["LLM_PORT"] = "8000"

print("3dSt_Coder + Ollama (Saul Instruct)")
print("=" * 40)
print("[OK] Using model: adrienbrault/saul-instruct-v1:q4_k_m")
print("[START] Starting 3dSt_Coder API server...")
print("[WEB] Web interface: http://localhost:8000")
print("[API] API docs: http://localhost:8000/docs")
print("\nPress Ctrl+C to stop")
print("=" * 40)

try:
    # Import and run
    from api.main import app
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

except KeyboardInterrupt:
    print("\n[STOP] Shutting down 3dSt_Coder...")
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    sys.exit(1)