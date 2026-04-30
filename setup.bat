@echo off
echo ================================================
echo  YVR Advanced Image Viewer - Setup
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================================
echo  Setup complete!
echo  Run:  venv\Scripts\activate && python main.py
echo ================================================
pause
