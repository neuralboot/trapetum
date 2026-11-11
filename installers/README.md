# Trapetum installers

Install the **resident inference server** on a machine with an NVIDIA GPU (Linux, Windows) or on
an Apple Silicon Mac. It serves a local ChatGPT-style web UI on `http://localhost:8088`, as a
background service on Linux and Windows.

The app is **chat-only**: it runs models that are **already compressed** (4-bit `.cbk`).
Compression is done ahead of time on our side, never on the user's machine, so the installer is
light (the `serve` binary plus the CUDA runtime, no Python or PyTorch). Models are installed from
the Models tab: the server downloads pre-compressed files from the Trapetum registry
(`cdn.neuralboot.com`) and checks `model.cbk` against the SHA-256 listed in the registry manifest
before serving it. The download buttons live on the web page (trapetum.neuralboot.com), not in the app.

## Requirements
- Linux or Windows: NVIDIA GPU + CUDA runtime (driver providing `libcudart`).
  macOS: Apple Silicon (M1 to M4), macOS 13 or later.
- Models directory: `~/.trapetum` on Linux and macOS. On Windows, `%LOCALAPPDATA%\Trapetum\models` of
  the account that runs the service: `C:\Windows\System32\config\systemprofile\AppData\Local\Trapetum\models`
  for the .msi (scheduled task as SYSTEM; only SYSTEM and administrators can read it, since
  `config.toml` holds the admin key and API tokens), the installing administrator's
  `%LOCALAPPDATA%` for the portable .zip. Each model is `<name>/model.cbk` + `tokenizer.json`
  + `config.json` + `meta.json`.

## Version numbers
Every published build gets a new build number N and the label `0.1.0-beta.N`. N is stamped into
the binary through `TRAPETUM_BUILD`; every 6 hours the server compares it with
`https://cdn.neuralboot.com/dist/latest.json`. The Windows MSI uses ProductVersion `0.1.N`
(MajorUpgrade compares only the first three fields) and the file name
`trapetum-0.1.0-beta.N-x64.msi`; 0.1.0-beta.1 shipped as `trapetum-0.1.0-x64.msi` (ProductVersion 0.1.0).

## Linux (.tar.gz + systemd)
```bash
# on a Linux x86_64 box with the CUDA toolkit and Rust
TRAPETUM_BUILD=2 ./installers/build-linux-bundle.sh 0.1.0-beta.2
# -> installers/dist/trapetum-linux-0.1.0-beta.2.tar.gz, published as dist/trapetum-linux.tar.gz

# on the target machine
tar xzf trapetum-linux.tar.gz
sudo ./trapetum-linux/install-linux.sh        # installs + enables the systemd service
```
Manage: `systemctl status|stop|restart trapetum`, logs `journalctl -u trapetum -f`,
remove `sudo ./uninstall-linux.sh`. Linux never updates itself: the server shows a notice.

## macOS (Apple Silicon)
```bash
TRAPETUM_BUILD=2 ./installers/build-macos-bundle.sh 0.1.0-beta.2
# -> installers/dist/trapetum-macos-arm64-0.1.0-beta.2.tar.gz, published as dist/trapetum-macos-arm64.tar.gz
```
The bundle installs `~/.trapetum-bin/trapetum`; start it with `~/.trapetum-bin/trapetum ~/.trapetum`.
No background service and no automatic update on macOS: the server shows a notice.

## Windows (.msi + scheduled task)
Build on the Windows build box (VS BuildTools, CUDA 12.6, Rust, WiX v3):
```powershell
powershell -File build-windows.ps1 -Build 2        # add -Sign once a certificate exists (CODE-SIGNING.md)
```
The MSI installs `serve.exe` and the CUDA runtime under `Program Files\Trapetum` and registers a
scheduled task `Trapetum` that runs as SYSTEM at startup. The portable zip uses
`install-windows.ps1` (elevated PowerShell). The one-line installer
(`irm get.neuralboot.com/install.ps1 | iex`) installs the MSI named in `latest.json` after
checking its SHA-256.

With automatic updates on (the default), an installed Windows build downloads the MSI named in
`latest.json`, checks its SHA-256 and installs it silently. Publish in this order: upload the new
MSI, test the upgrade from the previous build as SYSTEM, then publish `latest.json`.

## Optional: live carbon data
Set the grid zone (a country code such as `FR`, or an Electricity Maps zone such as
`US-CAL-CISO`) in Settings. Without it, energy and CO2 figures use a world average, labelled as
such. With a zone and an Electricity Maps token (Settings, or `TRAPETUM_CARBON_TOKEN` in
`trapetum.service`), the server fetches the live intensity of that zone. Since 0.1.0-beta.2 the
server never geolocates its IP address.
