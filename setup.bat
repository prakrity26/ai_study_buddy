@echo off
REM ================================================================
REM  setup.bat  --  Full first-time setup for Kathford AI Study Buddy
REM  Usage: Double-click or run from Command Prompt
REM
REM  What this does (in order):
REM    1. Install uv  (Python package manager)
REM    2. Install Ollama  (local LLM server)
REM    3. Pull the LLM model
REM    4. Create .env from .env.example
REM    5. Install Python dependencies  (uv sync)
REM    6. Index all study material  (ingest)
REM    7. Start the app
REM ================================================================

cd /d "%~dp0"
SET MODEL=qwen2.5:7b
SET PORT=8501

echo.
echo   ============================================
echo     Kathford AI Study Buddy -- Full Setup
echo   ============================================
echo.

REM ── 1. Install uv ─────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo [!] uv installed. Please CLOSE this window and run setup.bat again.
    pause
    exit /b 0
)
for /f "tokens=*" %%v in ('uv --version') do echo [setup] %%v

REM ── 2. Install Ollama ─────────────────────────────────────────
where ollama >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing Ollama via winget...
    winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
    echo.
    echo [!] Ollama installed. Please CLOSE this window and run setup.bat again.
    pause
    exit /b 0
)
echo [setup] Ollama found

REM ── 3. Pull the LLM model ─────────────────────────────────────
echo [setup] Checking model: %MODEL%
ollama list | findstr /i "%MODEL%" >nul 2>&1
if errorlevel 1 (
    echo [setup] Pulling model: %MODEL%  (may take a few minutes)...
    ollama pull %MODEL%
) else (
    echo [setup] Model already present: %MODEL%
)
echo [setup] Model ready: %MODEL%

REM ── 4. Create .env ────────────────────────────────────────────
if exist ".env" (
    echo [setup] .env already exists -- skipping
) else (
    copy .env.example .env >nul
    echo [setup] .env created from .env.example
)

REM ── 5. Install Python dependencies ────────────────────────────
echo [setup] Installing Python dependencies...
uv sync
echo [setup] Dependencies installed

REM ── 6. Index all study material ───────────────────────────────
echo.
echo [setup] Indexing study material in Data/ ...
echo [warn]  First run downloads the embedding model (~400MB) -- this takes a while.
echo [warn]  Already-indexed subjects are skipped automatically on re-runs.
echo.
uv run python src/ingest.py --sem all
echo.
echo [setup] Indexing complete

REM ── 7. Start the app ──────────────────────────────────────────
echo.
echo   ============================================
echo     Starting Kathford AI Study Buddy
echo     URL: http://localhost:%PORT%
echo     Press Ctrl+C to stop
echo   ============================================
echo.

uv run streamlit run app.py ^
    --server.port=%PORT% ^
    --server.address=0.0.0.0 ^
    --server.headless=true ^
    --server.fileWatcherType=poll

pause
