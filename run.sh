#!/bin/bash
# run.sh - Run the API server

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "🚀 Starting Political Sentiment Analysis API..."
echo "📍 Server: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
echo "🔍 Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
