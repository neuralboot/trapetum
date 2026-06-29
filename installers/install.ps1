# Trapetum one-line installer for Windows.
#   irm get.neuralboot.com/install.ps1 | iex
# Self-elevates, downloads the .msi, installs the background service, starts on :8088.
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
  Write-Host "Trapetum needs administrator rights to install a service. A UAC prompt will appear..."
  Start-Process powershell -Verb RunAs -ArgumentList `
    '-NoProfile','-ExecutionPolicy','Bypass','-Command','irm https://get.neuralboot.com/install.ps1 | iex'
  return
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Write-Warning "nvidia-smi not found. Trapetum needs an NVIDIA GPU (Ampere/Ada/Hopper) with a recent driver."
}

$msi = Join-Path $env:TEMP "trapetum-0.1.0-x64.msi"
Write-Host "Downloading Trapetum installer..."
Invoke-WebRequest -UseBasicParsing "https://cdn.neuralboot.com/dist/trapetum-0.1.0-x64.msi" -OutFile $msi

Write-Host "Installing..."
$p = Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /qb /norestart" -Wait -PassThru
if ($p.ExitCode -ne 0) { Write-Error "Installer exited with code $($p.ExitCode)."; return }

Write-Host "Waiting for the server to come up..."
$up = $false
for ($i = 0; $i -lt 25; $i++) {
  try { if ((Invoke-WebRequest -UseBasicParsing "http://localhost:8088/" -TimeoutSec 3).StatusCode -eq 200) { $up = $true; break } } catch {}
  Start-Sleep -Seconds 1
}
Write-Host ""
if ($up) { Write-Host "Trapetum is installed and RUNNING at  http://localhost:8088" }
else     { Write-Host "Trapetum is installed. The server is starting; open http://localhost:8088 in a moment." }
Write-Host "  next   : add your first model in the Admin page  ->  http://localhost:8088/admin"
Write-Host "  manage : taskschd.msc (task 'Trapetum')"
Write-Host "  remove : Control Panel > Programs and Features"
# desktop shortcut so the user can reopen the app anytime
try {
  $desktop = [Environment]::GetFolderPath("Desktop")
  if ($desktop) {
    Set-Content -Path (Join-Path $desktop "Trapetum.url") -Encoding ASCII `
      -Value "[InternetShortcut]`r`nURL=http://localhost:8088`r`nIconIndex=0`r`n"
  }
} catch {}
# open the app in the default browser so the user sees it is installed + the next steps
Start-Process "http://localhost:8088"
