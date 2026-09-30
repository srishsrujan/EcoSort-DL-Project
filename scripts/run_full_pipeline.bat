@echo off
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run scripts\setup_windows.bat first.
  exit /b 1
)
.venv\Scripts\python.exe -m src.run_pipeline %*
