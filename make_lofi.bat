@echo off
REM Generate tracks and build one continuous lofi song.
REM Usage:  make_lofi.bat [hours] [style] [tracks]
setlocal
cd /d "%~dp0"

set HOURS=%1
set STYLE=%2
set TRACKS=%3
if "%HOURS%"=="" set HOURS=1
if "%STYLE%"=="" set STYLE=lofi
if "%TRACKS%"=="" set TRACKS=12

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

echo Building a %HOURS%-hour "%STYLE%" song from %TRACKS% generated tracks...
echo.
"%PY%" pipeline.py all --style %STYLE% --count %TRACKS% --track-minutes 3 --hours %HOURS% --spine 0.5
if errorlevel 1 goto :fail

echo.
"%PY%" pipeline.py check
echo.
echo Done. The MP3 and its tracklist are in the newest folder under runs\
echo To lofi songs you already own instead, use make_lofify.bat
pause
exit /b 0

:fail
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
