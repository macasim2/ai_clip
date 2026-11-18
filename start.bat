@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
python -m src.launcher %*
endlocal