@echo off
REM ============================================================
REM runme.bat
REM
REM Double-click launcher for the EuroSAT training + dashboard
REM pipeline. Trains the model (if needed) and opens the
REM Streamlit dashboard in your browser.
REM
REM Place this file in the project root:
REM   multimodal_rs_pipeline\runme.bat
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Multi-Modal Remote Sensing Pipeline - EuroSAT Dashboard
echo ============================================================
echo.

REM ------------------------------------------------------------
REM Step 1: Locate conda and activate the rs_pipeline environment.
REM CALL is required here -- without it, running a .bat-based
REM conda activation from inside another .bat file terminates
REM this script early instead of continuing to the next line.
REM ------------------------------------------------------------
echo [1/3] Activating conda environment "rs_pipeline"...
call conda activate rs_pipeline 2>nul
if errorlevel 1 (
    echo       Plain "conda activate" failed, trying full conda hook...
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" rs_pipeline 2>nul
)
if errorlevel 1 (
    echo.
    echo ERROR: Could not activate conda environment "rs_pipeline".
    echo   - Make sure Anaconda/Miniconda is installed.
    echo   - Make sure the environment exists: conda create -n rs_pipeline python=3.10
    echo   - Try running this from an "Anaconda Prompt" instead of double-clicking,
    echo     since conda is sometimes not on PATH in a plain double-click session.
    echo.
    pause
    exit /b 1
)
echo       Environment activated.
echo.

REM ------------------------------------------------------------
REM Step 2: Train the model only if results don't already exist.
REM Delete the results\ folder if you want to force a fresh run.
REM ------------------------------------------------------------
echo [2/3] Checking for existing training results...
if exist "results\training_history.json" (
    echo       Found existing results\training_history.json - skipping training.
    echo       Delete the "results" folder if you want to retrain from scratch.
) else (
    echo       No existing results found. Starting training...
    echo.
    python train_eurosat_cnn.py
    if errorlevel 1 (
        echo.
        echo ERROR: Training script failed. See the output above for details.
        echo.
        pause
        exit /b 1
    )
)
echo.

REM ------------------------------------------------------------
REM Step 3: Launch the Streamlit dashboard.
REM This call blocks (Streamlit keeps running) until you close
REM this window or press Ctrl+C, which is expected.
REM ------------------------------------------------------------
echo [3/3] Launching dashboard...
echo       Your browser should open automatically.
echo       Close this window (or press Ctrl+C) to stop the dashboard.
echo.
streamlit run dashboard.py

echo.
echo Dashboard closed.
pause