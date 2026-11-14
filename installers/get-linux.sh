#!/bin/sh
# Trapetum one-line installer:  curl -fsSL get.neuralboot.com | sh
# Published as s3://neuralboot-trapetum-models/install.sh (served at https://get.neuralboot.com/).
# Checks the bundle against its published SHA-256 before installing; works as root without sudo.
set -e
BUNDLE="https://cdn.neuralboot.com/dist/trapetum-linux.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
printf '\033[38;5;208m▸\033[0m Downloading Trapetum…\n'
curl -fsSL -o "$TMP/trapetum-linux.tar.gz" "$BUNDLE"
WANT="$(curl -fsSL "$BUNDLE.sha256" | cut -d' ' -f1)"
GOT="$(sha256sum "$TMP/trapetum-linux.tar.gz" | cut -d' ' -f1)"
if [ -z "$WANT" ] || [ "$WANT" != "$GOT" ]; then
  echo "The download does not match its published SHA-256; nothing was installed."
  exit 1
fi
tar xzf "$TMP/trapetum-linux.tar.gz" -C "$TMP"
if [ "$(id -u)" -eq 0 ]; then
  printf '\033[38;5;208m▸\033[0m Installing…\n'
  env TRAPETUM_ADMIN_PASS="${TRAPETUM_ADMIN_PASS:-}" bash "$TMP/trapetum-linux/install-linux.sh"
else
  command -v sudo >/dev/null 2>&1 || { echo "Run this as root, or install sudo."; exit 1; }
  printf '\033[38;5;208m▸\033[0m Installing (needs sudo)…\n'
  sudo env TRAPETUM_ADMIN_PASS="${TRAPETUM_ADMIN_PASS:-}" bash "$TMP/trapetum-linux/install-linux.sh"
fi
