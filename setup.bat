@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv" (
    py -3 -m venv .venv
)

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Virtual environment is missing Python.
    exit /b 1
)

"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r requirements.txt

echo Setup complete.

endlocal
