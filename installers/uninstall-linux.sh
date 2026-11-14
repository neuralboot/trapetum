#!/usr/bin/env bash
# Remove the Trapetum systemd service (keeps your models in ~/.trapetum).
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Please run with sudo: sudo /opt/trapetum/uninstall-linux.sh"; exit 1; }
if [ -d /run/systemd/system ]; then
  systemctl stop trapetum 2>/dev/null || true
  systemctl disable trapetum 2>/dev/null || true
  rm -f /etc/systemd/system/trapetum.service
  systemctl daemon-reload || true
else
  pkill -f '/opt/trapetum/serve' 2>/dev/null || true
fi
rm -rf /opt/trapetum
echo "Trapetum removed. Your models and settings in ~/.trapetum were kept."
