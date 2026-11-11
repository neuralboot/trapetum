#!/bin/sh
# Trapetum macOS (Apple Silicon) installer. Runs the Metal engine locally.
set -e
DEST="$HOME/.trapetum-bin"; mkdir -p "$DEST"
cp "$(dirname "$0")/trapetum" "$DEST/trapetum"; chmod +x "$DEST/trapetum"
mkdir -p "$HOME/.trapetum"
printf '\033[38;5;208m▸\033[0m Installed. Start it with:\n   %s/trapetum ~/.trapetum\n' "$DEST"
printf '   then open http://localhost:8088\n'
