@echo off
setlocal
title Tri-dexel Simulation

set "ENV_NAME=tridexel-occ"
set "PROJECT_DIR=%~dp0"
set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"

if not exist "%CONDA_BAT%" (
    echo Conda launcher not found:
    echo   %CONDA_BAT%
    echo.
    echo Open Anaconda Prompt and run:
    echo   conda activate %ENV_NAME%
    echo   cd /d "%PROJECT_DIR%"
    echo   python app.py
    pause
    exit /b 1
)

call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment: %ENV_NAME%
    echo.
    echo Create it first with:
    echo   conda create -n %ENV_NAME% -c conda-forge python=3.10 pythonocc-core numpy scipy scikit-image pyvista pyvistaqt vtk pyqt gmsh pytest
    pause
    exit /b 1
)

python -c "from PyQt6.QtWidgets import QApplication; import pyvista; import pyvistaqt; import qtpy; import vtk" >nul 2>&1
if errorlevel 1 (
    echo GUI dependencies are missing in environment: %ENV_NAME%
    echo Installing PyQt6 / PyVista dependencies ...
    python -m pip install PyQt6==6.7.1 PyQt6-Qt6==6.7.1 PyQt6-sip pyvista pyvistaqt qtpy vtk --disable-pip-version-check
    if errorlevel 1 (
        echo.
        echo Failed to install GUI dependencies.
        pause
        exit /b 1
    )
)

python -c "import numpy; import skimage; import gmsh" >nul 2>&1
if errorlevel 1 (
    echo Simulation dependencies are missing in environment: %ENV_NAME%
    echo Installing numerical / meshing dependencies ...
    python -m pip install numpy scikit-image gmsh --disable-pip-version-check
    if errorlevel 1 (
        echo.
        echo Failed to install simulation dependencies.
        pause
        exit /b 1
    )
)

python -c "from OCC.Core.STEPControl import STEPControl_Writer; print('OpenCascade OK')" >nul 2>&1
if errorlevel 1 (
    echo OpenCascade is not available in environment: %ENV_NAME%
    echo.
    echo Run:
    echo   conda activate %ENV_NAME%
    echo   conda install -c conda-forge pythonocc-core
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
echo Environment: %ENV_NAME%
echo Launching GUI ...
python app.py

if errorlevel 1 (
    echo.
    echo ERROR: app.py crashed. See message above.
    pause
)
