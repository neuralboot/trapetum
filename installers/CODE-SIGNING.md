# Windows code signing (Authenticode)

Unsigned artifacts trigger the SmartScreen "Publisher: Unknown" warning and hurt
install conversion. Signing `serve.exe` and the `.msi` with an Authenticode
certificate replaces that with a verified publisher name and, with an EV cert,
removes the warning immediately.

## Choosing a certificate (note: OV/EV require a registered legal entity)

Modern code-signing keys must live on hardware (HSM / token) or a cloud signing
service; plain `.pfx` files are no longer issued for public trust.

| Option | Cost | SmartScreen trust | Entity needed | Good for |
|---|---|---|---|---|
| **Azure Trusted Signing** (individual) | ~$10/mo | builds over time | no (individual ID) | automation now |
| **Certum** Open Source Code Signing (individual, SimplySign cloud) | ~€90/yr | builds over time | no (individual ID) | cheapest now |
| **OV** (Sectigo / SSL.com, cloud HSM) | ~$200-350/yr | builds over days/weeks | yes | after incorporation |
| **EV** (Sectigo / DigiCert / SSL.com, token/HSM) | ~$300-500/yr | **immediate** | yes | B2B, post-incorporation |

**Recommended path given the September incorporation:**
1. **Now (pre-incorporation):** get an **individual** cert (Azure Trusted Signing
   individual, or Certum) to start signing and building reputation. Early testers
   still click "Run anyway" until reputation accrues, but the publisher is named.
2. **September (once neuralboot SAS exists):** move to an **EV** cert under the
   company for instant SmartScreen trust, which matters for a B2B security product.

## Chosen path: Azure Trusted Signing (individual)

**What you do in Azure (one-time; I cannot do this, it needs your account + ID):**
1. In the Azure portal, create a **Trusted Signing account** (resource "Trusted Signing", a supported region, Basic tier ~$9.99/mo).
2. Run **Identity validation** on it, choose **Individual**, submit your ID (approval ~1-3 business days).
   - Caveat: individual public-trust validation availability depends on country. Confirm it is open for France; if not, fall back to a **Certum** individual cert (EU-friendly) now, or the EV/org path once neuralboot SAS exists.
3. Create a **Certificate Profile** of type **Public Trust** (the one SmartScreen trusts). Note the account name, profile name, and the account's region endpoint.
4. Assign the role **"Trusted Signing Certificate Profile Signer"** on the profile to the identity that signs: either your Azure user (then `az login` on the build box) or a service principal (then set `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET`).

**On the build box (already wired in this repo):**
1. `powershell -File setup-azure-signing.ps1` installs the signing dlib and prints the env vars to set.
2. Edit `trusted-signing.metadata.json`: set `CodeSigningAccountName`, `CertificateProfileName`, and `Endpoint` to your region (e.g. `https://weu.codesigning.azure.net` for West Europe).
3. Set `TRAPETUM_SIGN_METHOD=ats`, `ATS_DLIB=<printed path>`, `ATS_METADATA=...\trusted-signing.metadata.json`, then authenticate (`az login` or the service-principal env vars).
4. `powershell -File build-windows.ps1 -Sign` produces a signed + timestamped `serve.exe` and `.msi`.

Reputation note: individual certs earn SmartScreen trust over a few weeks of downloads; until then a "Run anyway" may still appear once. An EV cert (post-incorporation, under the SAS) removes it immediately.

## Wiring it into the build

`build-windows.ps1 -Sign` signs `serve.exe` before packaging and the `.msi` after
WiX, both via `sign-windows.ps1` (SHA-256 + RFC-3161 timestamp). Pick the backend
with `TRAPETUM_SIGN_METHOD`:

- **Hardware token / HSM in the machine store** (EV tokens, most OV):
  `setx TRAPETUM_SIGN_METHOD auto` then `powershell -File build-windows.ps1 -Sign`
- **Azure Trusted Signing:** install the signing dlib, then set
  `TRAPETUM_SIGN_METHOD=ats`, `ATS_DLIB=<path\Azure.CodeSigning.Dlib.dll>`,
  `ATS_METADATA=<path\metadata.json>` and run `build-windows.ps1 -Sign`.
- **.pfx (test only):** `TRAPETUM_SIGN_METHOD=pfx`, `TRAPETUM_PFX=<file>`,
  `TRAPETUM_PFX_PASS=<pass>`.

Verify any artifact with: `signtool verify /pa /v trapetum-0.1.0-x64.msi`

## Optional: sign the one-line installer

`install-windows.ps1` (the script behind `irm get.neuralboot.com/install.ps1 | iex`)
can also be Authenticode-signed with `Set-AuthenticodeSignature`, which lets
security-conscious users run it under the `AllSigned` execution policy and shows
your publisher on the script itself. Sign it with the same certificate after each
edit, then re-upload to `get.neuralboot.com/install.ps1`.
