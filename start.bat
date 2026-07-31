@echo off
REM ================================================================
REM  start.bat  --  Start the Kathford AI Study Buddy on Windows
REM  Usage: Double-click or run from Command Prompt
REM ================================================================

cd /d "%~dp0"

if not exist ".env" (
    echo ERROR: .env file not found.
    echo Copy .env.example to .env and edit it first.
    pause
    exit /b 1
)

echo ================================================
echo   Kathford AI Study Buddy
echo   URL: http://localhost:8501
echo   Press Ctrl+C to stop
echo ================================================
echo.

uv run streamlit run app.py ^
    --server.port=8501 ^
    --server.address=0.0.0.0 ^
    --server.headless=true ^
    --server.fileWatcherType=poll

pause
