@echo off
setlocal EnableExtensions
title Tri-dexel Setup

set "ENV_NAME=tridexel-occ"
set "PYTHON_VERSION=3.10"
set "PROJECT_DIR=%~dp0"
set "REQ_FILE=%PROJECT_DIR%requirements.txt"

echo ============================================================
echo Tri-dexel environment setup
echo Project: %PROJECT_DIR%
echo Environment: %ENV_NAME%
echo ============================================================
echo.

call :find_conda
if errorlevel 1 (
    echo ERROR: Conda was not found.
    echo.
    echo Install Anaconda or Miniconda first, then run setup.bat again.
    echo Recommended: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo Conda: %CONDA_BAT%
echo.

call "%CONDA_BAT%" env list | findstr /B /C:"%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment: %ENV_NAME%
    call "%CONDA_BAT%" create -y -n "%ENV_NAME%" -c conda-forge python=%PYTHON_VERSION% pip pythonocc-core
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create conda environment.
        pause
        exit /b 1
    )
) else (
    echo Environment already exists: %ENV_NAME%
)

echo.
echo Activating environment...
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate environment: %ENV_NAME%
    pause
    exit /b 1
)

set "QT_API=pyqt6"

echo.
echo Ensuring OpenCascade / pythonocc-core is installed...
python -c "from OCC.Core.STEPControl import STEPControl_Writer" >nul 2>&1
if errorlevel 1 (
    call "%CONDA_BAT%" install -y -n "%ENV_NAME%" -c conda-forge pythonocc-core
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install pythonocc-core from conda-forge.
        pause
        exit /b 1
    )
)

echo.
echo Installing Python dependencies from requirements.txt...
if not exist "%REQ_FILE%" (
    echo ERROR: requirements.txt was not found:
    echo   %REQ_FILE%
    pause
    exit /b 1
)

python -m pip install --upgrade pip --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

python -m pip install -r "%REQ_FILE%" --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install requirements.txt.
    pause
    exit /b 1
)

echo.
echo Verifying runtime dependencies...
python -c "import os; os.environ.setdefault('QT_API', 'pyqt6'); import numpy, pyvista, pyvistaqt, qtpy, vtk, gmsh, numba; from PyQt6.QtWidgets import QApplication; from skimage.measure import marching_cubes; from OCC.Core.STEPControl import STEPControl_Writer; print('Dependency check OK')"
if errorlevel 1 (
    echo.
    echo ERROR: Dependency verification failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete.
echo You can now double-click run.bat to start the application.
echo ============================================================
echo.
pause
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
