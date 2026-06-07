@echo off
setlocal
cd /d "%~dp0"
echo.
echo  ====================================
echo     Localia RAG - starting
echo  ====================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [X] Python not found.
  echo     Install Python 3.10+ from https://www.python.org/downloads/
  echo     Tick "Add python.exe to PATH" during setup, then run this file again.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [*] First run: setting up the environment ^(1-2 min^)...
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [X] Dependency install failed. Check your connection and run again.
    pause
    exit /b 1
  )
)

where ollama >nul 2>nul
if errorlevel 1 (
  echo [X] Ollama not found. Install it from https://ollama.com then run this file again.
  pause
  exit /b 1
)

echo [*] Checking models ^(a download may happen on first run - please wait^)...
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

echo.
echo [*] Starting Localia RAG... a page will open in your browser.
echo     To stop it: close this window.
echo.
".venv\Scripts\python.exe" app.py
pause
