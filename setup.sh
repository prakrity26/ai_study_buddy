#!/usr/bin/env bash
# ================================================================
#  setup.sh  --  Full first-time setup for Kathford AI Study Buddy
#  Usage:  chmod +x setup.sh && ./setup.sh
#
#  What this does (in order):
#    1. Install uv  (Python package manager)
#    2. Install Ollama  (local LLM server)
#    3. Start Ollama server
#    4. Pull the LLM model
#    5. Create .env from .env.example
#    6. Install Python dependencies  (uv sync)
#    7. Index all study material  (ingest)
#    8. Start the app
# ================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Configurable via environment ─────────────────────────────────
MODEL="${LLM_MODEL:-qwen2.5:7b}"
PORT="${STREAMLIT_SERVER_PORT:-8501}"

# ── Colors ───────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }
fail() { echo -e "${RED}[error]${NC} $1"; exit 1; }

echo ""
echo "  ============================================"
echo "    Kathford AI Study Buddy — Full Setup"
echo "  ============================================"
echo ""

# ── 1. Install uv ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    # Also try cargo path (some installs use it)
    export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v uv &>/dev/null || fail "uv not found after install. Open a new terminal and re-run."
log "uv $(uv --version)"

# ── 2. Install Ollama ────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
command -v ollama &>/dev/null || fail "ollama not found after install. Open a new terminal and re-run."
log "Ollama $(ollama --version 2>/dev/null || echo 'installed')"

# ── 3. Start Ollama server ───────────────────────────────────────
if ! pgrep -f "ollama serve" &>/dev/null; then
    log "Starting Ollama server in background..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    sleep 4
    log "Ollama server started (PID $OLLAMA_PID)"
else
    log "Ollama server already running"
fi

# ── 4. Pull the LLM model ────────────────────────────────────────
if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    log "Model already present: $MODEL"
else
    log "Pulling model: $MODEL  (may take a few minutes on first run)..."
    ollama pull "$MODEL"
fi
log "Model ready: $MODEL"

# ── 5. Create .env ───────────────────────────────────────────────
if [[ -f ".env" ]]; then
    log ".env already exists — skipping"
else
    cp .env.example .env
    log ".env created from .env.example"
    # Ensure model name matches what we just pulled
    sed -i "s|^LLM_MODEL=.*|LLM_MODEL=${MODEL}|" .env
fi

# ── 6. Install Python dependencies ──────────────────────────────
log "Installing Python dependencies (uv sync)..."
uv sync
log "Dependencies installed"

# ── 7. Index all study material ──────────────────────────────────
log "Indexing study material in Data/ ..."
warn "First run downloads the embedding model (~400MB) — this takes a while."
warn "Already-indexed subjects are skipped automatically on re-runs."
echo ""
uv run python src/ingest.py --sem all
echo ""
log "Indexing complete"

# ── 8. Start the app ────────────────────────────────────────────
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "  ============================================"
echo "    Starting Kathford AI Study Buddy"
echo "    Local:   http://localhost:${PORT}"
echo "    Network: http://${LOCAL_IP}:${PORT}"
echo "    Press Ctrl+C to stop"
echo "  ============================================"
echo ""

uv run streamlit run app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=poll
