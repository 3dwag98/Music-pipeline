@echo off
REM One-time setup for the local lofi pipeline on Windows.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install Python 3.11 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment...
  python -m venv .venv || goto :fail
)

echo Installing packages...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail

echo.
echo Checking this machine...
call ".venv\Scripts\python.exe" pipeline.py doctor

echo.
echo Setup finished.
echo   make_lofi.bat          makes a one-hour song
echo   .venv\Scripts\activate then use pipeline.py directly
pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
