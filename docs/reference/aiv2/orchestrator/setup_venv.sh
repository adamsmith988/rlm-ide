#!/bin/bash
# Orchestrator Virtual Environment Setup
# راه‌اندازی محیط مجازی جداگانه برای ارکستراتور

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🚀 Setting up Orchestrator virtual environment..."
echo "   Directory: $VENV_DIR"

# Check Python version
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3.10+ not found!"
    exit 1
fi

echo "   Using: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo "📦 Virtual environment already exists"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "To run the service:"
echo "   cd $SCRIPT_DIR"
echo "   uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload"
