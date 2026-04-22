@echo off
REM Research Assistant Startup Script for Windows
REM Double-click this file to start both servers

echo ====================================
echo Research Assistant - Quick Start
echo ====================================

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python startup script
python start_server.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed. Press any key to exit...
    pause > nul
)