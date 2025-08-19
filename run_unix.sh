#!/bin/bash

echo "========================================"
echo "    PriceWise Application Launcher"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found:"
python3 --version

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade pip
echo
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo
echo "📦 Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if .streamlit directory exists
if [ ! -d ".streamlit" ]; then
    echo
    echo "📁 Creating .streamlit configuration directory..."
    mkdir -p .streamlit
    echo "✅ Configuration directory created"
fi

# Check if secrets.toml exists
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo
    echo "⚠️  No OpenAI API key configured"
    echo "Please create .streamlit/secrets.toml with your API key"
    echo
    echo "Example content:"
    echo "[openai]"
    echo "api_key = \"your-api-key-here\""
    echo
    read -p "Press Enter to continue..."
fi

echo
echo "🚀 Starting PriceWise Application..."
echo
echo "📱 The app will open in your browser at: http://localhost:8501"
echo
echo "💡 To stop the app, press Ctrl+C in this terminal"
echo

# Start the application
streamlit run app/main.py --server.port=8501
