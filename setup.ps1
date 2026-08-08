$ErrorActionPreference = "Stop"

# Always work from the repository root, even if this script is launched elsewhere.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "=== bvar-energy environment setup ==="
Write-Host "Project root: $ProjectRoot"
Write-Host ""

$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $Requirements)) {
    throw "requirements.txt was not found at: $Requirements"
}

# Prefer the Windows Python launcher pinned to Python 3.11.
if (-not (Test-Path $PythonExe)) {
    Write-Host "[1/5] Creating .venv with Python 3.11..."

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & py -3.11 -m venv $VenvDir
    }
    else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $PythonCommand) {
            throw "Python was not found. Install Python 3.11 and rerun setup.ps1."
        }

        $Version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($Version.Trim() -ne "3.11") {
            throw "Python 3.11 is required, but 'python' points to Python $Version."
        }
        & python -m venv $VenvDir
    }
}
else {
    Write-Host "[1/5] Existing .venv found; keeping it."
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment creation failed: $PythonExe does not exist."
}

Write-Host "[2/5] Upgrading pip..."
& $PythonExe -m pip install --upgrade pip

Write-Host "[3/5] Installing requirements..."
& $PythonExe -m pip install -r $Requirements

Write-Host "[4/5] Registering the Jupyter kernel..."
& $PythonExe -m ipykernel install `
    --user `
    --name "bvar-energy" `
    --display-name "Python (bvar-energy)"

$KernelScript = Join-Path $ProjectRoot "scripts\configure_notebook_kernels.py"
if (Test-Path $KernelScript) {
    Write-Host "      Updating notebook kernelspec metadata..."
    & $PythonExe $KernelScript
}

Write-Host "[5/5] Verifying the environment..."
& $PythonExe -c @"
import sys
import numpy
import pandas
import scipy
import statsmodels
import matplotlib
import openpyxl
import ipykernel

print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("numpy             :", numpy.__version__)
print("pandas            :", pandas.__version__)
print("scipy             :", scipy.__version__)
print("statsmodels       :", statsmodels.__version__)
print("matplotlib        :", matplotlib.__version__)
print("openpyxl          :", openpyxl.__version__)
print("ipykernel         :", ipykernel.__version__)

if sys.version_info[:2] != (3, 11):
    raise SystemExit("ERROR: this project expects Python 3.11.")
"@

Write-Host ""
Write-Host "Environment ready."
Write-Host "VS Code interpreter:"
Write-Host "  $PythonExe"
Write-Host "Notebook kernel:"
Write-Host "  Python (bvar-energy)"
Write-Host ""
Write-Host "If VS Code was already open, run 'Developer: Reload Window' once."
