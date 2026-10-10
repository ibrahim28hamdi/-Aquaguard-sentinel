@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run START_WINDOWS.cmd to create the environment.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m unittest discover -s tests -v
pause
