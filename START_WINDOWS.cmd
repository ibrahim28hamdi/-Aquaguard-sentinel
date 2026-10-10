@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>&1
  if errorlevel 1 (
    echo Python launcher is missing. Install Python 3.10+ and try again.
    pause
    exit /b 1
  )
  py -m venv .venv
  if errorlevel 1 goto error
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error
exit /b 0
:error
echo Startup failed. See the error above.
pause
exit /b 1
