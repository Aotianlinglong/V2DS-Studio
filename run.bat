@echo off
rem V2DS Studio launcher - bundled Python is in this folder python\
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%python\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    pause
    exit /b 1
)

cd /d "%ROOT%"
set "PYTHONHOME=%ROOT%python"
set "PYTHONPATH=%ROOT%"
"%PY%" main.py
set "EXIT=%ERRORLEVEL%"
if not "%EXIT%"=="0" (
    echo [ERROR] V2DS Studio exited with code %EXIT%
    pause
)
exit /b %EXIT%
