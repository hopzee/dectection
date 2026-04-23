@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Virtual environment is missing Python. Run setup.bat first.
    exit /b 1
)

"%PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit is missing. Run setup.bat first.
    exit /b 1
)

echo Starting Streamlit...
"%PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501

endlocal
