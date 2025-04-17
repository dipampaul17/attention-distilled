#!/bin/bash
# Script to run the FastAPI translation service directly

# Make sure we're in the virtual environment
source venv/bin/activate

# Install FastAPI and Uvicorn if not already installed
pip install fastapi==0.110 uvicorn[standard]==0.29

# Kill any existing uvicorn processes
pkill -f "uvicorn serve:app" || true

# Start the FastAPI service
uvicorn serve:app --host 0.0.0.0 --port 8000
