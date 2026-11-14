# Trapetum service setup, called by the MSI (deferred, elevated). Headless: no prompts.
# Registers a native Windows scheduled task that runs serve.exe at startup and starts it now.
# Binds to localhost with no admin password by default; the user sets a password later in /admin.
$ErrorActionPreference = "Stop"
$Here       = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = $Here
$Models     = Join-Path $env:LOCALAPPDATA "Trapetum\models"
# Upgrade: keep the models folder already in use. LOCALAPPDATA differs between the user who installed by hand
# and SYSTEM, which runs the automatic update; without this an update would lose the models and the settings.
# Order: the folder recorded by a previous install (HKLM\SOFTWARE\Trapetum, written from 0.1.0-beta.3 on), the
# folder of the task still registered, then an existing folder of an earlier install (the most recently used).
$known = $null
try { $known = (Get-ItemProperty "HKLM:\SOFTWARE\Trapetum" -ErrorAction Stop).ModelsDir } catch {}
if (-not $known) {
  try {
    $old = (Get-ScheduledTask -TaskName "Trapetum" -ErrorAction Stop).Actions[0].Arguments
    if ($old -match 'serve\.exe"\s+"([^"]+)"\s+8088') { $known = $matches[1] }
  } catch {}
}
if (-not $known) {
  $cands = @(Get-ChildItem "C:\Users\*\AppData\Local\Trapetum\models\config.toml", "$env:WINDIR\System32\config\systemprofile\AppData\Local\Trapetum\models\config.toml" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
  if ($cands.Count -gt 0) { $known = $cands[0].DirectoryName }
}
if ($known -and (Test-Path $known)) { $Models = $known }
New-Item -Path "HKLM:\SOFTWARE\Trapetum" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Trapetum" -Name ModelsDir -Value $Models
New-Item -ItemType Directory -Force -Path $Models | Out-Null

$cfgPath = Join-Path $Models "config.toml"
if (-not (Test-Path $cfgPath)) {
  $cfg = @"
port = 8088
bind = "127.0.0.1"
admin_key = ""
api_tokens = []
default_model = ""
cors_origins = ""
max_tokens_cap = 4096
rate_limit_rpm = 0
log_prompts = false
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
