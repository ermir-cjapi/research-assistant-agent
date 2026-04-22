#!/usr/bin/env python3
"""
Backend-only startup script for the Research Assistant.

Usage: python start_backend.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    print("[STARTUP] Research Assistant Backend")
    print("=" * 40)
    
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    
    # Check if backend exists
    if not backend_dir.exists():
        print("[ERROR] Backend directory not found")
        return 1
    
    # Check if venv exists
    venv_dir = backend_dir / "venv"
    if not venv_dir.exists():
        print("[ERROR] Virtual environment not found at backend/venv")
        return 1
    
    # Get Python executable
    if sys.platform == 'win32':
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
    
    print(f"[BACKEND] Using Python: {python_exe}")
    print("[BACKEND] Starting server...")
    print("[BACKEND] URL: http://localhost:8000")
    print("[BACKEND] API Docs: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    try:
        # Start the server
        subprocess.run([
            str(python_exe), "server.py"
        ], cwd=str(backend_dir), check=True)
        
    except KeyboardInterrupt:
        print("\n[STOPPED] Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())