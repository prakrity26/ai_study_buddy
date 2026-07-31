#!/usr/bin/env bash
# ================================================================
#  start.sh  --  Start the Kathford AI Study Buddy on Linux/WSL
#  Usage: chmod +x start.sh && ./start.sh
# ================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f ".env" ]]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and edit it first."
    exit 1
fi

echo "================================================"
echo "  Kathford AI Study Buddy"
echo "  URL: http://localhost:8501"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo

uv run streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=poll
