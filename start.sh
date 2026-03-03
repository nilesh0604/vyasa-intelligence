#!/bin/bash

# Vyasa Intelligence - Startup Script

echo "=== Vyasa Intelligence Startup ==="

# Check if virtual environment exists
if [ ! -d "venv-python311" ]; then
    echo "Error: Python 3.11 virtual environment not found!"
    echo "Please run setup first:"
    echo "  /opt/homebrew/bin/python3.11 -m venv venv-python311"
    echo "  source venv-python311/bin/activate"
    echo "  pip install -e ."
    exit 1
fi

# Activate virtual environment
echo "Activating Python 3.11 virtual environment..."
source venv-python311/bin/activate

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Warning: Ollama is not running!"
    echo "Please start Ollama: ollama serve &"
    echo "Then pull the model: ollama pull llama3.2"
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
