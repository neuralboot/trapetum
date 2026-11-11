# Trapetum service setup, called by the MSI (deferred, elevated). Headless: no prompts.
# Registers a native Windows scheduled task that runs serve.exe at startup and starts it now.
# Binds to localhost with no admin password by default; the user sets a password later in /admin.
$ErrorActionPreference = "Stop"
$Here       = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = $Here
$Models     = Join-Path $env:LOCALAPPDATA "Trapetum\models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

$cfgPath = Join-Path $Models "config.toml"
if (-not (Test-Path $cfgPath)) {
  $cfg = @"
port = 8088
bind = "127.0.0.1"
admin_key = ""
api_tokens = []
default_model = ""
cors_origins = "*"
max_tokens_cap = 4096
rate_limit_rpm = 0
log_prompts = true
carbon_token = ""
"@
  Set-Content -Path $cfgPath -Value $cfg -Encoding UTF8
}

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
exit 0
