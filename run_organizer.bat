@echo off
title Travel Photo Organizer
echo ===============================
echo   Travel Photo Organizer
echo ===============================
echo.

:: Always run from the folder this .bat lives in
cd /d "%~dp0"

:: Step 1: Install Python packages (only needed first time)
echo Checking and installing dependencies...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Couldn't install dependencies automatically.
    echo Make sure Python and pip are installed, then run this manually:
    echo     pip install -r requirements.txt
    pause
    exit /b
)

echo Dependencies OK.
echo.

:: Step 2: Run the Qt UI
python ui_app_qt.py

:: Step 3: Auto-close if successful
if %errorlevel% equ 0 (
    exit
)

:: Optional: keep open if an error occurs
echo.
echo ⚠️  Something went wrong. Please check the message above.
pause
