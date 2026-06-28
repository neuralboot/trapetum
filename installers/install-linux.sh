#!/usr/bin/env bash
# Trapetum Linux installer — installs the local inference server as a systemd service.
# Usage:  sudo ./install-linux.sh [path-to-serve-binary]
# The service starts on boot, restarts on failure, and serves the web UI on http://localhost:8088
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN_SRC="${1:-$HERE/serve}"
USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
INSTALL_DIR=/opt/trapetum
MODELS_DIR="$HOME_DIR/.trapetum"

[ "$(id -u)" -eq 0 ] || { echo "Please run with sudo: sudo ./install-linux.sh"; exit 1; }
[ -f "$BIN_SRC" ] || { echo "serve binary not found at $BIN_SRC"; exit 1; }

VER="$(cat "$HERE/VERSION" 2>/dev/null || echo '0.1.0-beta')"
echo "Trapetum installer  ·  v$VER"
echo "  user        : $USER_NAME"
echo "  binary      : $INSTALL_DIR/serve"
echo "  models dir  : $MODELS_DIR"
command -v nvidia-smi >/dev/null 2>&1 \
  && echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  || echo "  GPU         : WARNING — nvidia-smi not found; Trapetum needs an NVIDIA GPU + CUDA runtime."

# install binary + models dir
mkdir -p "$INSTALL_DIR"
install -m 0755 "$BIN_SRC" "$INSTALL_DIR/serve"
mkdir -p "$MODELS_DIR"
chown "$USER_NAME" "$MODELS_DIR"

# set the admin password that locks the /admin settings page
CFG="$MODELS_DIR/config.toml"
ADMIN_PASS="${TRAPETUM_ADMIN_PASS:-}"
if [ -z "$ADMIN_PASS" ] && [ -t 0 ]; then
  echo ""
  read -rsp "Set an admin password for the settings page (Enter to keep local-only access): " ADMIN_PASS
  echo ""
fi
ADMIN_PASS="${ADMIN_PASS//\"/}"   # strip any double quotes for safe TOML
BIND_ADDR="127.0.0.1"
[ -n "$ADMIN_PASS" ] && BIND_ADDR="0.0.0.0"   # password set -> safe to expose on the network
cat > "$CFG" <<EOF
port = 8088
bind = "$BIND_ADDR"
admin_key = "$ADMIN_PASS"
api_tokens = []
default_model = ""
cors_origins = "*"
max_tokens_cap = 4096
rate_limit_rpm = 0
log_prompts = true
carbon_token = ""
EOF
chown "$USER_NAME" "$CFG"
[ -n "$ADMIN_PASS" ] && echo "Admin password set. /admin is now password-protected." \
  || echo "No password set. /admin is reachable only from this machine (localhost)."

# render + install the systemd unit
sed -e "s#__USER__#$USER_NAME#g" \
    -e "s#__HOME__#$HOME_DIR#g" \
    -e "s#__BIN__#$INSTALL_DIR/serve#g" \
    -e "s#__MODELS__#$MODELS_DIR#g" \
    "$HERE/trapetum.service" > /etc/systemd/system/trapetum.service

# drop any old @reboot cron launcher (superseded by the service) and stop any running server
( crontab -u "$USER_NAME" -l 2>/dev/null | grep -v run-serve.sh ) | crontab -u "$USER_NAME" - 2>/dev/null || true
pkill -9 -f 'release/serve' 2>/dev/null || true
pkill -9 -f '/opt/trapetum/serve' 2>/dev/null || true
sleep 2

systemctl daemon-reload
systemctl enable trapetum >/dev/null 2>&1
systemctl restart trapetum

sleep 3
echo ""
echo "Installed. Trapetum is running at  http://localhost:8088"
echo "  status : systemctl status trapetum"
echo "  logs   : journalctl -u trapetum -f"
echo "  stop   : sudo systemctl stop trapetum     (start / restart likewise)"
echo "  remove : sudo ./uninstall-linux.sh"
