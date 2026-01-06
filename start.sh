#!/bin/bash

# Activate venv if it exists
if [[ -d ".venv" ]]; then
	source .venv/bin/activate
fi

echo "🚀 Starting NEXUS System..."

# Start Backend
echo "Starting Backend (Port 8000)..."
uvicorn nexus.api.server:app --port 8000 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd ui || exit
npm run dev &
FRONTEND_PID=$!

# Handle shutdown
cleanup() {
	echo ""
	echo "🛑 Shutting down NEXUS..."
	if ps -p "${BACKEND_PID}" >/dev/null; then
		kill "${BACKEND_PID}"
	fi
	if ps -p "${FRONTEND_PID}" >/dev/null; then
		kill "${FRONTEND_PID}"
	fi
	exit
}

trap cleanup SIGINT

# Keep script running to maintain the trap
wait
