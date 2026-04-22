#!/usr/bin/env python3
"""
Complete Research Assistant Startup Script

This script starts both the backend and frontend servers:
1. Activates virtual environment in backend/venv
2. Starts Python server from backend directory
3. Starts React frontend with npm run dev
4. Opens browser tabs for both servers

Usage: python start_server.py
"""
import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path
from threading import Thread


def check_dependencies():
    """Check if required directories and files exist."""
    project_root = Path(__file__).parent
    
    # Check backend
    backend_dir = project_root / "backend"
    venv_dir = backend_dir / "venv"
    server_file = backend_dir / "server.py"
    
    if not backend_dir.exists():
        print("[ERROR] Backend directory not found")
        return False
    
    if not venv_dir.exists():
        print("[ERROR] Virtual environment not found at backend/venv")
        print("[TIP] Create with: cd backend && python -m venv venv")
        return False
        
    if not server_file.exists():
        print("[ERROR] server.py not found in backend directory")
        return False
    
    # Check frontend
    frontend_dir = project_root / "frontend"
    package_json = frontend_dir / "package.json"
    
    if not frontend_dir.exists():
        print("[ERROR] Frontend directory not found")
        return False
        
    if not package_json.exists():
        print("[ERROR] package.json not found in frontend directory")
        return False
    
    print("[OK] All required files and directories found")
    return True


def start_backend():
    """Start the Python backend server."""
    print("\n[BACKEND] Starting Python server...")
    
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    
    # Use venv Python
    if os.name == 'nt':  # Windows
        python_exe = backend_dir / "venv" / "Scripts" / "python.exe"
    else:  # Unix/Linux/MacOS
        python_exe = backend_dir / "venv" / "bin" / "python"
    
    try:
        # Start the server from backend directory
        process = subprocess.Popen([
            str(python_exe), "server.py"
        ], cwd=str(backend_dir))
        
        print(f"[BACKEND] Server starting with PID: {process.pid}")
        print("[BACKEND] URL: http://localhost:8000")
        
        # Wait a moment for server to start
        time.sleep(3)
        return process
        
    except Exception as e:
        print(f"[ERROR] Failed to start backend: {e}")
        return None


def start_frontend():
    """Start the React frontend server."""
    print("\n[FRONTEND] Starting React frontend...")
    
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    try:
        # Check if node_modules exists
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print("[FRONTEND] Installing dependencies...")
            subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True, shell=True)
        
        # Start the development server
        process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=str(frontend_dir), shell=True)
        
        print(f"[FRONTEND] Server starting with PID: {process.pid}")
        print("[FRONTEND] URL: http://localhost:3000")
        
        # Wait a moment for server to start
        time.sleep(5)
        return process
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start frontend: {e}")
        return None
    except FileNotFoundError:
        print("[ERROR] npm not found. Please install Node.js and npm")
        return None


def open_browser():
    """Open browser tabs for both servers."""
    print("\n[BROWSER] Opening browser tabs...")
    
    # Wait for servers to be ready
    time.sleep(3)
    
    try:
        # Open backend API docs
        webbrowser.open("http://localhost:8000/")
        time.sleep(1)
        
        # Open frontend
        webbrowser.open("http://localhost:3000/")
        print("[BROWSER] Opened tabs for both servers")
        
    except Exception as e:
        print(f"[WARNING] Could not open browser: {e}")
        print("[INFO] Manual URLs:")
        print("  Backend:  http://localhost:8000")
        print("  Frontend: http://localhost:3000")


def main():
    """Main startup orchestration."""
    print("=" * 60)
    print("[STARTUP] Research Assistant - Full Stack Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Start backend in a separate thread
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            return 1
        
        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            if backend_process:
                backend_process.terminate()
            return 1
        
        # Open browser (optional)
        try:
            browser_thread = Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        except Exception as e:
            print(f"[WARNING] Could not start browser thread: {e}")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Both servers started successfully!")
        print("=" * 60)
        print("[BACKEND]  Python API: http://localhost:8000")
        print("[FRONTEND] React UI:   http://localhost:3000")
        print()
        print("[FEATURES]")
        print("  - Upload documents via the UI")
        print("  - Ask questions about your documents")
        print("  - Search Wikipedia for general knowledge")  
        print("  - Perform calculations")
        print("  - View API docs at http://localhost:8000/docs")
        print()
        print("Press Ctrl+C to stop both servers")
        print("=" * 60)
        
        # Wait for user interruption
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            pass
            
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping servers...")
        
    finally:
        # Clean shutdown
        if backend_process:
            print("[SHUTDOWN] Stopping Python server...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
                
        if frontend_process:
            print("[SHUTDOWN] Stopping React server...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)  
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        print("[SHUTDOWN] All servers stopped")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[CANCELLED] Startup cancelled by user")
        sys.exit(0)