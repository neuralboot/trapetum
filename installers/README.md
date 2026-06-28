# Trapetum installers

Install the **resident inference server** on a machine with an NVIDIA GPU. It runs as a
background service and serves a local ChatGPT-style web UI on `http://localhost:8088`.

The app is **chat-only**: it runs models that are **already compressed** (4-bit `.cbk`).
Compression is done ahead of time, not on the user's machine — so the installer is light
(just the `serve` binary + CUDA runtime, no Python/PyTorch). The compression story and the
download buttons live on the web front page (neuralboot.com/trapetum), not in the app.

## Requirements
- NVIDIA GPU + CUDA runtime (driver providing `libcudart`)
- One or more compressed models in the models directory (`~/.trapetum` on Linux,
  `%LOCALAPPDATA%\Trapetum\models` on Windows), each as `<name>/model.cbk` + `tokenizer.json`
  + `config.json` + `meta.json`.

## Linux (.tar.gz + systemd)
```bash
# build the bundle on a Linux box that already built the serve binary
(cd runtime && cargo build --release --bin serve)
./installers/build-linux-bundle.sh            # -> installers/dist/trapetum-linux-<ver>.tar.gz

# on the target machine
tar xzf trapetum-linux-<ver>.tar.gz
sudo ./trapetum-linux/install-linux.sh        # installs + enables the systemd service
```
Manage: `systemctl status|stop|restart trapetum`, logs `journalctl -u trapetum -f`,
remove `sudo ./uninstall-linux.sh`.

## Windows (.exe + service via NSSM)
Build `serve.exe` on Windows (`cargo build --release --bin serve`), place it next to
`nssm.exe` and the scripts, then in an **elevated** PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File install-windows.ps1
```
Manage from `services.msc` or `nssm start|stop|restart Trapetum`.

## Building both from CI
`serve.exe` cannot be built from macOS/Linux — use GitHub Actions with `windows-latest`
(Windows installer) and `ubuntu-latest` (Linux bundle) runners on tag, publishing the
artifacts to GitHub Releases. The web page's download buttons then point at the release URLs.

## Optional: live carbon data
Set a free [ElectricityMaps](https://www.electricitymaps.com) token so the server reports
the real-time grid carbon intensity at its location:
- Linux: uncomment `Environment=TRAPETUM_CARBON_TOKEN=...` in `trapetum.service`
- Windows: `nssm set Trapetum AppEnvironmentExtra TRAPETUM_CARBON_TOKEN=...`
Without a token it uses the geo-located grid's recent average (never a hard-coded global value).
