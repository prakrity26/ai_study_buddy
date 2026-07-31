#!/usr/bin/env bash
# ================================================================
#  ingest_all.sh  --  Index all study material in Data/
#  Usage: chmod +x ingest_all.sh && ./ingest_all.sh
# ================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f ".env" ]]; then
    echo "ERROR: .env file not found. Copy .env.example to .env first."
    exit 1
fi

echo "Indexing all study material in Data/ ..."
echo "This may take 10-30 minutes on first run (downloads embedding model)."
echo

uv run python src/ingest.py --sem all

echo
echo "Done. Start the app with: ./start.sh"
