@echo off
REM ================================================================
REM  ingest_all.bat  --  Index all study material in Data/
REM  Run this after adding new books, before starting the app.
REM ================================================================

cd /d "%~dp0"

if not exist ".env" (
    echo ERROR: .env file not found. Copy .env.example to .env first.
    pause
    exit /b 1
)

echo Indexing all study material in Data/ ...
echo This may take 10-30 minutes on first run (downloads embedding model).
echo.

uv run python src/ingest.py --sem all

echo.
echo Done. Start the app with: start.bat
pause
