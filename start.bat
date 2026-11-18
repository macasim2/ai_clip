@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10-3.14 from python.org.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create .venv. Check your Python installation.
        pause
        exit /b 1
    )
)

set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)"
if errorlevel 1 (
    echo [ERROR] Unsupported Python version:
    "%PY%" -c "import sys; print(sys.version)"
    echo Install Python 3.12 (recommended) or 3.14 and recreate .venv.
    pause
    exit /b 1
)

"%PY%" -c "import os,sys; sys.exit(0 if (not os.path.exists('.venv/.deps_ok') or os.path.getmtime('requirements.txt') > os.path.getmtime('.venv/.deps_ok')) else 1)"
if not errorlevel 1 (
    echo [2/3] Installing dependencies...
    "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. See the messages above.
        pause
        exit /b 1
    )
    type nul > ".venv\.deps_ok"
)

echo [3/3] Starting AI Clip Studio...
"%PY%" -m src.launcher %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
