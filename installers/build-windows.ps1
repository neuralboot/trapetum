# Trapetum Windows build + optional Authenticode signing.
# Runs on the Windows build box (VS BuildTools + CUDA 12.6 + Rust + WiX v3 installed).
#
#   powershell -File build-windows.ps1                 # build only (unsigned)
#   powershell -File build-windows.ps1 -Sign           # build + sign serve.exe and the .msi
#
# Signing backend is selected inside sign-windows.ps1 via env TRAPETUM_SIGN_METHOD (auto|ats|pfx).
param(
  [switch]$Sign,
  [string]$Src   = "C:\build\runtime",                                   # cargo crate (has cuda/, src/)
  [string]$Inst  = "C:\build\installers",                                # this folder (wxs + scripts)
  [string]$Cuda  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
  [string]$Wix   = "C:\wix",                                             # candle.exe / light.exe
  [string]$Ver   = "0.1.0",
  [int]$Build    = 1,                                                    # monotonic build number, drives auto-update
  [string]$MsiUrl = "https://cdn.neuralboot.com/dist/trapetum-0.1.0-x64.msi"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. build serve.exe (CUDA 12.6)
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" | Out-Null
$env:CUDA_PATH = $Cuda
$env:CUDA_ARCH = "sm_80"
$env:PATH = "$env:PATH;$Cuda\bin;C:\ProgramData\chocolatey\lib\rust-ms\tools\bin;C:\ProgramData\chocolatey\bin"
$env:TRAPETUM_BUILD = "$Build"          # stamped into the binary; the updater compares this to the manifest
Push-Location $Src
cargo build --release --bin serve
Pop-Location
$serve = Join-Path $Src "target\release\serve.exe"
if (-not (Test-Path $serve)) { throw "build failed: $serve missing" }

# 2. sign serve.exe BEFORE packaging (so the signed binary is what ships)
if ($Sign) { & (Join-Path $here "sign-windows.ps1") -Files $serve }

# 3. stage files for the .msi + zip
$stage = Join-Path $Inst "stage"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item $serve (Join-Path $stage "serve.exe") -Force
Copy-Item (Join-Path $Cuda "bin\cudart64_12.dll") (Join-Path $stage "cudart64_12.dll") -Force

# 4. build the .msi (WiX v3)
Push-Location $Inst
& "$Wix\candle.exe" -nologo -arch x64 -ext WixUtilExtension trapetum.wxs -o trapetum.wixobj
& "$Wix\light.exe"  -nologo -ext WixUtilExtension -ext WixUIExtension -sw1076 trapetum.wixobj -o "trapetum-$Ver-x64.msi"
Pop-Location
$msi = Join-Path $Inst "trapetum-$Ver-x64.msi"
if (-not (Test-Path $msi)) { throw "WiX build failed: $msi missing" }

# 5. sign the .msi
if ($Sign) { & (Join-Path $here "sign-windows.ps1") -Files $msi }

# 6. portable zip (signed serve.exe + runtime + scripts)
$pkg = Join-Path $Inst "pkg"
Remove-Item -Recurse -Force $pkg -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $pkg | Out-Null
Copy-Item (Join-Path $stage "serve.exe") $pkg -Force
Copy-Item (Join-Path $stage "cudart64_12.dll") $pkg -Force
Copy-Item (Join-Path $Inst "install-windows.ps1") $pkg -Force
Copy-Item (Join-Path $Inst "uninstall-windows.ps1") $pkg -Force
$zip = Join-Path $Inst "trapetum-windows-x64.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$pkg\*" -DestinationPath $zip -Force

# 7. update the auto-update manifest (publishing latest.json to the CDN triggers installed apps to update)
$sha = (Get-FileHash $msi -Algorithm SHA256).Hash.ToLower()
$manifest = Join-Path $Inst "latest.json"
[ordered]@{
  build    = $Build
  version  = "$Ver-beta.$Build"
  url      = $MsiUrl
  sha256   = $sha
  notes    = "Trapetum Windows build $Build"
  released = (Get-Date -Format "yyyy-MM-dd")
} | ConvertTo-Json | Set-Content -Path $manifest -Encoding UTF8

Write-Host "`nBuilt (build $Build):"
Write-Host "  $msi"
Write-Host "  $zip"
Write-Host "  $manifest"
Write-Host "`nPublish to ship the update: upload the .msi to $MsiUrl, then latest.json to"
Write-Host "  https://cdn.neuralboot.com/dist/latest.json  (installed apps with auto-update poll it and reinstall)."
if (-not $Sign) { Write-Warning "UNSIGNED build. Pass -Sign once a code-signing cert is configured (see CODE-SIGNING.md)." }
