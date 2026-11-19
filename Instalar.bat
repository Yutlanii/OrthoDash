@echo off
rem Instala el entorno de Python (solo la primera vez). Requiere Python 3.10 a 3.13.
cd /d "%~dp0"
set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY ( py -%%V -c "1" >nul 2>&1 && set "PY=py -%%V" )
)
if not defined PY (
  echo No se encontro Python 3.10-3.13. Instalalo desde https://www.python.org/downloads/
  pause
  exit /b 1
)
echo Usando %PY%
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install --retries 10 --timeout 120 -r requirements.txt
if errorlevel 1 (
  echo Error instalando dependencias. Revisa tu conexion e intenta de nuevo.
  pause
  exit /b 1
)
echo.
rem Componente GPU: solo se descarga si hay tarjeta NVIDIA. Si falla, el programa funciona igual con la CPU.
".venv\Scripts\python.exe" -m fotogrametria --instalar-gpu
echo.
rem Acceso directo con el icono del programa.
powershell -NoProfile -Command "$a=(New-Object -ComObject WScript.Shell).CreateShortcut('%~dp0OrthoDash.lnk'); $a.TargetPath='%~dp0Ejecutar.bat'; $a.WorkingDirectory='%~dp0'; $a.IconLocation='%~dp0fotogrametria\interfaz\web\logo.ico'; $a.WindowStyle=7; $a.Save()"
echo Instalacion completa. Ahora abre "OrthoDash".
pause
