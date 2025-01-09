#!/usr/bin/bash
# This stops all processes when the script is stopped
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

ollama serve &

source .venv/bin/activate

export SEMADB_API_KEY=
python3 main.py
