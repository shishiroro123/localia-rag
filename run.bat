@echo off
setlocal
cd /d "%~dp0"
echo.
echo  ====================================
echo     Localia RAG - demarrage
echo  ====================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [X] Python introuvable.
  echo     Installe Python 3.10+ depuis https://www.python.org/downloads/
  echo     Coche bien "Add python.exe to PATH" pendant l'installation, puis relance ce fichier.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [*] Premiere utilisation : installation de l'environnement ^(1-2 min^)...
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [X] L'installation des dependances a echoue. Verifie ta connexion et relance.
    pause
    exit /b 1
  )
)

where ollama >nul 2>nul
if errorlevel 1 (
  echo [X] Ollama introuvable. Installe-le depuis https://ollama.com puis relance ce fichier.
  pause
  exit /b 1
)

echo [*] Verification des modeles ^(un telechargement est possible la 1re fois - sois patient^)...
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

echo.
echo [*] Lancement de Localia RAG... une page va s'ouvrir dans ton navigateur.
echo     Pour arreter : ferme cette fenetre.
echo.
".venv\Scripts\python.exe" app.py
pause
