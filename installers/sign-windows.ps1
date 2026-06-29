# Authenticode signing for Trapetum Windows artifacts (serve.exe + the .msi).
# Backend-agnostic: works with a hardware token / HSM (auto), Azure Trusted Signing, or a .pfx.
# SHA-256 digest + RFC-3161 timestamp (so signatures stay valid after the cert expires).
#
#   powershell -File sign-windows.ps1 -Files serve.exe,trapetum-0.1.0-x64.msi
#
# Method is chosen by -Method (or env TRAPETUM_SIGN_METHOD):
#   auto   : cert in the machine store or a connected EV token   (signtool /a)          [default]
#   ats    : Azure Trusted Signing (env: ATS_DLIB, ATS_METADATA)
#   pfx    : a .pfx file (env: TRAPETUM_PFX, TRAPETUM_PFX_PASS)   [legacy / test only]
param(
  [Parameter(Mandatory = $true)][string[]]$Files,
  [string]$Method = $(if ($env:TRAPETUM_SIGN_METHOD) { $env:TRAPETUM_SIGN_METHOD } else { "auto" }),
  [string]$TimestampUrl = "http://timestamp.acs.microsoft.com"
)
$ErrorActionPreference = "Stop"

# locate signtool.exe from the Windows SDK
$signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName | Select-Object -Last 1 -ExpandProperty FullName
if (-not $signtool) { throw "signtool.exe not found. Install the Windows 10/11 SDK (Signing Tools)." }
Write-Host "signtool: $signtool"

foreach ($f in $Files) {
  if (-not (Test-Path $f)) { throw "file not found: $f" }
  $common = @("sign", "/fd", "SHA256", "/tr", $TimestampUrl, "/td", "SHA256", "/v")
  switch ($Method) {
    "auto" { $args = $common + @("/a", $f) }
    "pfx"  {
      if (-not $env:TRAPETUM_PFX) { throw "set TRAPETUM_PFX (and TRAPETUM_PFX_PASS)" }
      $args = $common + @("/f", $env:TRAPETUM_PFX, "/p", $env:TRAPETUM_PFX_PASS, $f)
    }
    "ats"  {
      if (-not $env:ATS_DLIB -or -not $env:ATS_METADATA) { throw "set ATS_DLIB and ATS_METADATA for Azure Trusted Signing" }
      $args = $common + @("/dlib", $env:ATS_DLIB, "/dmdf", $env:ATS_METADATA, $f)
    }
    default { throw "unknown -Method '$Method' (use auto|ats|pfx)" }
  }
  Write-Host "signing $f ..."
  & $signtool @args
  if ($LASTEXITCODE -ne 0) { throw "signtool failed on $f (exit $LASTEXITCODE)" }
}

Write-Host "`nverifying signatures..."
foreach ($f in $Files) { & $signtool verify /pa /v $f }
Write-Host "`nAll files signed and verified."
