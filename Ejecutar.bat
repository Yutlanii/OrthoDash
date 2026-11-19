@echo off
rem Abre la ventana de OrthoDash.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Primero ejecuta Instalar.bat
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m fotogrametria
