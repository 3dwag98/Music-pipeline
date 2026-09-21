@echo off
REM Turn songs you already own into lofi versions, then into one long song.
REM Usage:  make_lofify.bat "C:\path\to\your songs" [preset] [hours]
setlocal
cd /d "%~dp0"

set SRC=%~1
set PRESET=%~2
set HOURS=%~3
if "%SRC%"=="" (
  echo Drag a folder of songs onto this file, or run:
  echo   make_lofify.bat "C:\path\to\your songs" [preset] [hours]
  echo.
  echo Presets: classic, slowed, study, sleep, tape, instrumental
  pause
  exit /b 1
)
if "%PRESET%"=="" set PRESET=classic
if "%HOURS%"=="" set HOURS=1

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

echo.
echo These must be songs you own, are licensed to use, or that are public domain.
echo A lofi edit of someone else's record is still their record.
echo.
set /p OK="Type YES to continue: "
if /i not "%OK%"=="YES" (
  echo Cancelled.
  pause
  exit /b 1
)

"%PY%" pipeline.py lofify "%SRC%" --preset %PRESET% --i-own-this || goto :fail
"%PY%" pipeline.py song --hours %HOURS% || goto :fail
"%PY%" pipeline.py check

echo.
echo Done. The song and its tracklist are in the newest folder under runs\
pause
exit /b 0

:fail
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
