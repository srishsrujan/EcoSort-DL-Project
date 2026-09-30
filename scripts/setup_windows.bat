@echo off
setlocal

echo EcoSort AI - Windows environment setup
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Failed to create the virtual environment. Make sure Python 3.11 is installed.
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  exit /b 1
)

echo.
echo Environment ready.
echo Activate with: .venv\Scripts\activate.bat
echo Then run: python -m src.run_pipeline
endlocal
