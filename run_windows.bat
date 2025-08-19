@echo off
echo ========================================
echo    PriceWise Application Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Check if virtual environment exists
if not exist ".venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo.
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Check if .streamlit directory exists
if not exist ".streamlit" (
    echo.
    echo 📁 Creating .streamlit configuration directory...
    mkdir .streamlit
    echo ✅ Configuration directory created
)

REM Check if secrets.toml exists
if not exist ".streamlit\secrets.toml" (
    echo.
    echo ⚠️  No OpenAI API key configured
    echo Please create .streamlit\secrets.toml with your API key
    echo.
    echo Example content:
    echo [openai]
    echo api_key = "your-api-key-here"
    echo.
    pause
)

echo.
echo 🚀 Starting PriceWise Application...
echo.
echo 📱 The app will open in your browser at: http://localhost:8501
echo.
echo 💡 To stop the app, press Ctrl+C in this window
echo.

REM Start the application
streamlit run app/main.py --server.port=8501

pause
