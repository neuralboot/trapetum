#!/bin/sh
# Trapetum macOS (Apple Silicon) one-line installer:
#   curl -fsSL get.neuralboot.com/mac | sh
# Published as s3://neuralboot-trapetum-models/mac. Checks the bundle against its published SHA-256.
set -e
BUNDLE="https://cdn.neuralboot.com/dist/trapetum-macos-arm64.tar.gz"
TMP="$(mktemp -d)"; DEST="$HOME/.trapetum-bin"
trap 'rm -rf "$TMP"' EXIT
printf '\033[38;5;208m▸\033[0m Downloading Trapetum for Apple Silicon…\n'
curl -fsSL -o "$TMP/trapetum-macos.tar.gz" "$BUNDLE"
WANT="$(curl -fsSL "$BUNDLE.sha256" | cut -d' ' -f1)"
GOT="$(shasum -a 256 "$TMP/trapetum-macos.tar.gz" | cut -d' ' -f1)"
if [ -z "$WANT" ] || [ "$WANT" != "$GOT" ]; then
  echo "The download does not match its published SHA-256; nothing was installed."
  exit 1
fi
tar xzf "$TMP/trapetum-macos.tar.gz" -C "$TMP"
mkdir -p "$DEST" "$HOME/.trapetum"
cp "$TMP/trapetum-macos/trapetum" "$DEST/trapetum"; chmod +x "$DEST/trapetum"
printf '\033[38;5;208m▸\033[0m Installed. Start it with:\n   %s/trapetum ~/.trapetum\n   then open http://localhost:8088\n' "$DEST"
