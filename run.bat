@echo off
setlocal EnableExtensions
title Tri-dexel NC Machining Simulator

set "ENV_NAME=tridexel-occ"
set "PROJECT_DIR=%~dp0"

call :find_conda
if errorlevel 1 (
    echo ERROR: Conda was not found.
    echo.
    echo Run setup.bat after installing Anaconda or Miniconda.
    echo.
    pause
    exit /b 1
)

call "%CONDA_BAT%" env list | findstr /B /C:"%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda environment was not found: %ENV_NAME%
    echo.
    echo Double-click setup.bat first. It will create the environment and install dependencies.
    echo.
    pause
    exit /b 1
)

call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment: %ENV_NAME%
    echo.
    echo Double-click setup.bat to repair the environment.
    echo.
    pause
    exit /b 1
)

set "QT_API=pyqt6"

python -c "import os; os.environ.setdefault('QT_API', 'pyqt6'); import numpy, pyvista, pyvistaqt, qtpy, vtk, gmsh, numba; from PyQt6.QtWidgets import QApplication; from skimage.measure import marching_cubes; from OCC.Core.STEPControl import STEPControl_Writer" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Some dependencies are missing in environment: %ENV_NAME%
    echo.
    echo Double-click setup.bat to install or repair the environment.
    echo.
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
exit /b 0

:find_conda
set "CONDA_BAT="

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"

if not defined CONDA_BAT (
    for /f "delims=" %%I in ('conda info --base 2^>nul') do (
        if exist "%%I\condabin\conda.bat" set "CONDA_BAT=%%I\condabin\conda.bat"
    )
)

if not defined CONDA_BAT exit /b 1
exit /b 0
