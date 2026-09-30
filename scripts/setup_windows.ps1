$ErrorActionPreference = 'Stop'

Write-Host 'EcoSort AI - Windows environment setup' -ForegroundColor Green

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    throw 'Python launcher (py) was not found. Install Python 3.11 and try again.'
}

& py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ''
Write-Host 'Environment ready.' -ForegroundColor Green
Write-Host 'Activate with: .\.venv\Scripts\Activate.ps1'
Write-Host 'Then run: python -m src.run_pipeline'
