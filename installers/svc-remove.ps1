# Trapetum service teardown, called by the MSI on uninstall (deferred, elevated). Headless.
$ErrorActionPreference = "SilentlyContinue"
Stop-ScheduledTask     -TaskName "Trapetum" 2>$null
Unregister-ScheduledTask -TaskName "Trapetum" -Confirm:$false 2>$null
# best-effort: stop a running serve.exe
Get-Process serve -ErrorAction SilentlyContinue | Stop-Process -Force 2>$null
exit 0
