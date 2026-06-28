#!/usr/bin/env bash
# Remove the Trapetum systemd service (keeps your compressed models in ~/.trapetum).
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Please run with sudo: sudo ./uninstall-linux.sh"; exit 1; }
systemctl stop trapetum 2>/dev/null || true
systemctl disable trapetum 2>/dev/null || true
rm -f /etc/systemd/system/trapetum.service
systemctl daemon-reload
rm -rf /opt/trapetum
echo "Trapetum service removed. Your compressed models in ~/.trapetum were kept."
