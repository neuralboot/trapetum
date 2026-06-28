# Trapetum Windows uninstaller. Run in an ELEVATED PowerShell:
#   powershell -ExecutionPolicy Bypass -File uninstall-windows.ps1
$ErrorActionPreference = "SilentlyContinue"
$InstallDir = Join-Path $env:ProgramFiles "Trapetum"
Stop-ScheduledTask       -TaskName "Trapetum" 2>$null
Unregister-ScheduledTask -TaskName "Trapetum" -Confirm:$false 2>$null
Get-Process serve -ErrorAction SilentlyContinue | Stop-Process -Force 2>$null
Remove-Item -Recurse -Force $InstallDir 2>$null
Write-Host "Trapetum removed. Models under %LOCALAPPDATA%\Trapetum were kept."
