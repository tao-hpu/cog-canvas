#!/bin/bash

# CogCanvas Web Backend Startup Script
# Starts the FastAPI backend on port 3701

echo "Starting CogCanvas Web Backend..."
echo "Port: 3701"
echo "API Docs: http://localhost:3701/docs"
echo ""

# Navigate to backend directory
cd "$(dirname "$0")"

# Start the server
uvicorn main:app --reload --port 3701 --host 0.0.0.0
