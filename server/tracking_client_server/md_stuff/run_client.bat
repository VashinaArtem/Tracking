@echo off
REM MiniCPM-V Tracking Client - Quick Start Script for Windows
REM This script activates virtual environment and runs the client

echo ======================================
echo MiniCPM-V Tracking Client
echo ======================================
echo.

REM Check if virtual environment exists
if not exist "venv_client\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run the following commands first:
    echo   python -m venv venv_client
    echo   venv_client\Scripts\activate
    echo   pip install -r client_requirements.txt
    echo.
    pause
    exit /b 1
)

REM Get server URL from user
set /p SERVER_URL="Enter server URL (e.g., http://192.168.1.100:8000): "

if "%SERVER_URL%"=="" (
    echo [ERROR] Server URL is required!
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call venv_client\Scripts\activate.bat

REM Run the client
echo [2/2] Starting tracking client...
echo.
echo Connecting to: %SERVER_URL%
echo.
python tracking_client.py --server %SERVER_URL%

REM Deactivate on exit
deactivate
pause
