# Trapetum Windows installer (zip path). Installs the local inference server as a
# native scheduled task. Run in an ELEVATED PowerShell:
#   powershell -ExecutionPolicy Bypass -File install-windows.ps1
# Requires: NVIDIA GPU + CUDA runtime. Ships serve.exe + cudart64_12.dll.
$ErrorActionPreference = "Stop"
$Here       = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Join-Path $env:ProgramFiles "Trapetum"
$Models     = Join-Path $env:LOCALAPPDATA "Trapetum\models"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Write-Error "Please run this installer in an elevated (Administrator) PowerShell."; exit 1 }

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Write-Warning "nvidia-smi not found — Trapetum needs an NVIDIA GPU + CUDA runtime."
}

New-Item -ItemType Directory -Force -Path $InstallDir, $Models | Out-Null
Copy-Item (Join-Path $Here "serve.exe")         (Join-Path $InstallDir "serve.exe")         -Force
Copy-Item (Join-Path $Here "cudart64_12.dll")   (Join-Path $InstallDir "cudart64_12.dll")   -Force

# set the admin password that locks the /admin settings page
$Pass = Read-Host "Set an admin password for the settings page (Enter to keep local-only access)"
$Pass = $Pass -replace '"',''
$Bind = if ($Pass) { "0.0.0.0" } else { "127.0.0.1" }
$cfg = @"
port = 8088
bind = "$Bind"
admin_key = "$Pass"
api_tokens = []
default_model = ""
cors_origins = "*"
max_tokens_cap = 4096
rate_limit_rpm = 0
log_prompts = true
carbon_token = ""
"@
Set-Content -Path (Join-Path $Models "config.toml") -Value $cfg -Encoding UTF8
if ($Pass) { Write-Host "Admin password set. /admin is now password-protected." }
else { Write-Host "No password set. /admin is reachable only from this machine (localhost)." }

$exe = Join-Path $InstallDir "serve.exe"
$log = Join-Path $InstallDir "serve.log"
$arg = "/c `"`"$exe`" `"$Models`" 8088 > `"$log`" 2>&1`""
$action    = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $arg
$trigger   = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName "Trapetum" -Action $action -Trigger $trigger `
  -Principal $principal -Settings $settings -Description "Local 4-bit LLM inference server (neuralboot.com/trapetum)" -Force | Out-Null
Start-ScheduledTask -TaskName "Trapetum"

Write-Host ""
Write-Host "Installed. Trapetum is running at  http://localhost:8088"
Write-Host "  manage : taskschd.msc   (task 'Trapetum')   or   Start-ScheduledTask Trapetum"
Write-Host "  remove : powershell -ExecutionPolicy Bypass -File uninstall-windows.ps1"
