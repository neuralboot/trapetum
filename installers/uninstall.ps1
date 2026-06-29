# Trapetum one-line uninstaller for Windows.
#   irm get.neuralboot.com/uninstall.ps1 | iex
# Self-elevates, stops the service, removes it (MSI or portable), and cleans the folder.
$ErrorActionPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
  Write-Host "Trapetum uninstall needs administrator rights. A UAC prompt will appear..."
  Start-Process powershell -Verb RunAs -ArgumentList `
    '-NoProfile','-ExecutionPolicy','Bypass','-Command','irm https://get.neuralboot.com/uninstall.ps1 | iex'
  return
}

Write-Host "Stopping Trapetum..."
Stop-ScheduledTask       -TaskName "Trapetum" 2>$null
Unregister-ScheduledTask -TaskName "Trapetum" -Confirm:$false 2>$null
Get-Process serve -ErrorAction SilentlyContinue | Stop-Process -Force 2>$null

# if it was installed via the .msi, remove that product properly (clears Add/Remove Programs)
$roots = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
         "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
Get-ChildItem $roots -ErrorAction SilentlyContinue | ForEach-Object {
  $p = Get-ItemProperty $_.PSPath
  if ($p.DisplayName -match "Trapetum") {
    Write-Host "Removing MSI product $($p.DisplayName)..."
    Start-Process msiexec.exe -ArgumentList "/x $($_.PSChildName) /qn /norestart" -Wait
  }
}

# remove any leftover files (runtime logs the MSI does not track, or a portable/zip install)
$dir = Join-Path $env:ProgramFiles "Trapetum"
if (Test-Path $dir) { Remove-Item -Recurse -Force $dir 2>$null }

Write-Host ""
if (Test-Path $dir) { Write-Warning "Some files remain in $dir (a file may be in use). Reboot and delete the folder." }
else { Write-Host "Trapetum is fully removed." }
Write-Host "Your downloaded models under %LOCALAPPDATA%\Trapetum were kept. Delete that folder to remove them too."
