# One-time setup of Azure Trusted Signing on the Windows build box.
# Installs the signing dlib that signtool uses, then prints the env vars to set.
# Run AFTER you have created a Trusted Signing account + certificate profile in Azure.
param(
  [string]$Dir = "C:\trusted-signing",
  [string]$PkgVersion = ""   # empty = latest
)
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$pkg = "Microsoft.Trusted.Signing.Client"
New-Item -ItemType Directory -Force -Path "$Dir\pkg" | Out-Null

$url = if ($PkgVersion) { "https://www.nuget.org/api/v2/package/$pkg/$PkgVersion" }
       else            { "https://www.nuget.org/api/v2/package/$pkg" }
Write-Host "Downloading $pkg ..."
Invoke-WebRequest -UseBasicParsing $url -OutFile "$Dir\$pkg.zip"
Expand-Archive "$Dir\$pkg.zip" "$Dir\pkg" -Force

$dll = Get-ChildItem "$Dir\pkg" -Recurse -Filter "Azure.CodeSigning.Dlib.dll" |
       Where-Object { $_.FullName -match "x64" } | Select-Object -First 1 -ExpandProperty FullName
if (-not $dll) { throw "Azure.CodeSigning.Dlib.dll (x64) not found in the package" }

Write-Host ""
Write-Host "Trusted Signing dlib installed."
Write-Host "Set these before building with -Sign:"
Write-Host "  setx TRAPETUM_SIGN_METHOD ats"
Write-Host "  setx ATS_DLIB `"$dll`""
Write-Host "  setx ATS_METADATA `"$Dir\trusted-signing.metadata.json`""
Write-Host ""
Write-Host "Auth: sign in once with `az login` (Azure CLI), OR set a service principal:"
Write-Host "  setx AZURE_TENANT_ID <tenant>; setx AZURE_CLIENT_ID <appId>; setx AZURE_CLIENT_SECRET <secret>"
Write-Host "The SP / your user needs the role 'Trusted Signing Certificate Profile Signer' on the profile."
