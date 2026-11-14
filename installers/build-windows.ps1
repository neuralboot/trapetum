# Trapetum Windows build + optional Authenticode signing.
# Runs on the Windows build box (VS BuildTools + CUDA 12.6 + Rust + WiX v3 installed).
#
#   powershell -File build-windows.ps1 -Build 2                 # build only (unsigned)
#   powershell -File build-windows.ps1 -Build 2 -Sign           # build + sign serve.exe and the .msi
#
# -Build is the monotonic build number: it is stamped into serve.exe (the updater compares it
# with latest.json), it sets the MSI ProductVersion 0.1.<Build> (MajorUpgrade compares only the
# first three fields; 0.1.0-beta.1 shipped as 0.1.0) and it names the file, so a published
# installer is never overwritten.
# Signing backend is selected inside sign-windows.ps1 via env TRAPETUM_SIGN_METHOD (auto|ats|pfx).
param(
  [switch]$Sign,
  [Parameter(Mandatory = $true)][int]$Build,
  [string]$Src   = "C:\build\runtime",                                   # cargo crate (has cuda/, src/)
  [string]$Inst  = "C:\build\installers",                                # this folder (wxs + scripts)
  [string]$Cuda  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
  [string]$Wix   = "C:\wix",                                             # candle.exe / light.exe
  [string]$Ver   = "0.1.0",
  [string]$MsiUrl = ""
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Build -lt 2) { throw "-Build must be 2 or more: build 1 (MSI 0.1.0) is already published" }
# A build number is used once: two MSIs with the same ProductVersion install side by side.
try {
  $published = (Invoke-RestMethod -UseBasicParsing "https://cdn.neuralboot.com/dist/latest.json").build
  if ($Build -le [int]$published) { throw "-Build $Build is not above the published build $published" }
} catch [System.Net.WebException] { Write-Warning "could not read the published latest.json; check the build number by hand" }
$Label   = "$Ver-beta.$Build"                     # e.g. 0.1.0-beta.2
$MsiVer  = "0.1.$Build"                           # MSI ProductVersion
$MsiName = "trapetum-$Label-x64.msi"
if (-not $MsiUrl) { $MsiUrl = "https://cdn.neuralboot.com/dist/$MsiName" }

# 1. build serve.exe (CUDA 12.6); no CUDA_ARCH, so build.rs emits the portable multi-arch fatbin
# vcvars64.bat only changes the environment of the cmd.exe that runs it: import that environment here,
# so nvcc finds cl.exe and build.rs finds lib.exe
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] } }
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe not found after vcvars64.bat: install the VS 2022 C++ build tools" }
$env:CUDA_PATH = $Cuda
Remove-Item Env:CUDA_ARCH -ErrorAction SilentlyContinue
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
& "$Wix\candle.exe" -nologo -arch x64 -ext WixUtilExtension "-dProductVersion=$MsiVer" trapetum.wxs -o trapetum.wixobj
if ($LASTEXITCODE -ne 0) { throw "candle failed ($LASTEXITCODE)" }
& "$Wix\light.exe"  -nologo -ext WixUtilExtension -ext WixUIExtension -sw1076 trapetum.wixobj -o $MsiName
if ($LASTEXITCODE -ne 0) { throw "light failed ($LASTEXITCODE)" }
Pop-Location
$msi = Join-Path $Inst $MsiName
if (-not (Test-Path $msi)) { throw "WiX build failed: $msi missing" }

# 5. sign the .msi
if ($Sign) { & (Join-Path $here "sign-windows.ps1") -Files $msi }

# 6. portable zip (serve.exe + runtime + scripts)
$pkg = Join-Path $Inst "pkg"
Remove-Item -Recurse -Force $pkg -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $pkg | Out-Null
Copy-Item (Join-Path $stage "serve.exe") $pkg -Force
Copy-Item (Join-Path $stage "cudart64_12.dll") $pkg -Force
Copy-Item (Join-Path $Inst "install-windows.ps1") $pkg -Force
Copy-Item (Join-Path $Inst "uninstall-windows.ps1") $pkg -Force
Set-Content -Path (Join-Path $pkg "VERSION") -Value $Label -Encoding ASCII
$zip = Join-Path $Inst "trapetum-windows-x64.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$pkg\*" -DestinationPath $zip -Force

# 7. the auto-update manifest. Publishing latest.json makes every installed Windows build with
#    automatic updates download and install this MSI: upload the MSI first, test the upgrade
#    (previous build to this one, as SYSTEM), and only then publish latest.json.
$sha = (Get-FileHash $msi -Algorithm SHA256).Hash.ToLower()
$manifest = Join-Path $Inst "latest.json"
[ordered]@{
  build    = $Build
  version  = $Label
  url      = $MsiUrl
  sha256   = $sha
  notes    = "Trapetum $Label"
  released = (Get-Date -Format "yyyy-MM-dd")
} | ConvertTo-Json | ForEach-Object { [IO.File]::WriteAllText($manifest, $_, (New-Object System.Text.UTF8Encoding $false)) }   # no byte-order mark: serde_json rejects it

Write-Host "`nBuilt $Label (MSI ProductVersion $MsiVer):"
Write-Host "  $msi"
Write-Host "  $zip"
Write-Host "  $manifest"
Write-Host "`nPublish: upload the .msi to $MsiUrl, test the upgrade, then latest.json to"
Write-Host "  https://cdn.neuralboot.com/dist/latest.json  (installed apps with automatic updates poll it every 6 h)."
if (-not $Sign) { Write-Warning "UNSIGNED build. Pass -Sign once a code-signing cert is configured (see CODE-SIGNING.md)." }
