#!/usr/bin/env python3
"""
Simple startup script for the unified research assistant.

Usage: python start_unified.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("[STARTUP] Starting Unified Research Assistant")
    print("=" * 50)
    print("Features:")
    print("  - LangChain + LangGraph + LangSmith + RAG")
    print("  - Single clean implementation")
    print("  - Document upload via UI")
    print("  - Multi-source Q&A")
    print()
    print("[SERVER] Server: http://localhost:8000")
    print("[FRONTEND] Frontend: http://localhost:3000")
    print("[DOCS] Docs: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    # Start the unified server
    try:
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Server stopped")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\n[TIP] Make sure you have:")
        print("  1. Set OPENAI_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()