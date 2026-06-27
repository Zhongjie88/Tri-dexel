@echo off
title Tri-dexel Simulation

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+
    pause & exit /b 1
)

echo Installing dependencies ...
python -m pip install pyvista pyvistaqt "PyQt6==6.7.1" "PyQt6-Qt6==6.7.1" PyQt6-sip qtpy scikit-image numpy gmsh pytest --quiet --disable-pip-version-check

echo Launching GUI ...
cd /d "%~dp0"
python app.py

if errorlevel 1 (
    echo.
    echo ERROR: app.py crashed. See message above.
    pause
)