# install-and-test.ps1
# Run as Administrator
$ErrorActionPreference = 'Stop'

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Err($m){ Write-Host "[ERROR] $m" -ForegroundColor Red }

Write-Info "Step 1: Check for git..."
try {
    $gitVer = & git --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Git already installed: $gitVer"
        $gitInstalled = $true
    } else { $gitInstalled = $false }
} catch {
    $gitInstalled = $false
}

if (-not $gitInstalled) {
    Write-Info "Attempting to download Git for Windows installer..."
    $tempInstaller = Join-Path $env:TEMP "Git-64-bit.exe"
    $url = "https://github.com/git-for-windows/git/releases/latest/download/Git-64-bit.exe"

    try {
        # Ensure TLS 1.2
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $tempInstaller -UseBasicParsing -TimeoutSec 120
        if (Test-Path $tempInstaller) {
            Write-Info "Installer downloaded to $tempInstaller. Running silent install..."
            Start-Process -FilePath $tempInstaller -ArgumentList "/VERYSILENT","/NORESTART" -Wait -NoNewWindow
            Start-Sleep -Seconds 2
            Write-Info "Install finished, verifying git..."
            $gitVer = & git --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Info "Git installed: $gitVer"
                $gitInstalled = $true
            } else {
                Write-Err "Git not found after installer. You may need to open a NEW PowerShell session or add Git to PATH."
                $gitInstalled = $false
            }
        } else {
            Write-Err "Download didn't produce an installer file."
            $gitInstalled = $false
        }
    } catch {
        Write-Err "Automatic download/install failed: $($_.Exception.Message)"
        $gitInstalled = $false
    }
}

if (-not $gitInstalled) {
    Write-Host ""
    Write-Host "Automatic installation failed or network is blocked. Please manually download and install Git for Windows:" -ForegroundColor Yellow
    Write-Host "  https://git-scm.com/download/win" -ForegroundColor Green
    Write-Host "After installing, re-open PowerShell and re-run this script to continue."
    exit 1
}

# Step 2: Initialize local git repo and commit scaffold
Write-Info "Step 2: Initialize git repo and commit scaffold"
Push-Location "F:\GitHub\localLLM"
try {
    if (-not (Test-Path ".git")) {
        git init
        Write-Info "Created git repo."
    } else {
        Write-Info "Git repo already initialized."
    }
    git add .
    git commit -m "chore(scaffold): initial project scaffold per First.txt"
    if ($LASTEXITCODE -ne 0) { Write-Info "Nothing to commit or commit failed (maybe already committed)." }
} catch {
    Write-Err "Git operations failed: $($_.Exception.Message)"
    Pop-Location
    exit 1
}
Pop-Location

# Step 3: Check Python and run tests if Python exists
Write-Info "Step 3: Check for Python"
try {
    $py = & python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Python present: $py"
        $pythonPresent = $true
    } else {
        $pythonPresent = $false
    }
} catch {
    $pythonPresent = $false
}

if (-not $pythonPresent) {
    Write-Host ""
    Write-Host "Python not found in PATH. To run tests locally, install Python 3.12+ and re-run this script." -ForegroundColor Yellow
    Write-Host "Download: https://www.python.org/downloads/windows/" -ForegroundColor Green
    exit 0
}

# Use venv and pip to run tests
Write-Info "Creating venv and running pytest"
Push-Location "F:\GitHub\localLLM"
try {
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    $activate = ".\.venv\\Scripts\\Activate.ps1"
    if (Test-Path $activate) {
        # Run pytest in a child process to ensure activation
        & powershell -NoProfile -ExecutionPolicy Bypass -Command {
            Set-Location 'F:\GitHub\localLLM'
            . '.\.venv\Scripts\Activate.ps1'
            python -m pip install --upgrade pip
            pip install pytest pytest-asyncio -q
            pytest -q
        }
    } else {
        Write-Err "Virtualenv activation script not found: $activate"
    }
} catch {
    Write-Err "Running tests failed: $($_.Exception.Message)"
    Pop-Location
    exit 1
}
Pop-Location

Write-Info "Script completed."