#!/usr/bin/env bash
# Package a distributable Trapetum Linux installer bundle (.tar.gz).
# Run on a Linux box that has already built the serve binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN="$ROOT/runtime/target/release/serve"
[ -f "$BIN" ] || { echo "Build first:  (cd runtime && cargo build --release --bin serve)"; exit 1; }

VER="${1:-$(date +%Y.%m.%d)}"
STAGE="$HERE/dist/trapetum-linux"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$BIN" "$STAGE/serve"
cp "$HERE/install-linux.sh" "$HERE/uninstall-linux.sh" "$HERE/trapetum.service" "$STAGE/"
chmod +x "$STAGE/"*.sh "$STAGE/serve"

TARBALL="$HERE/dist/trapetum-linux-$VER.tar.gz"
tar -C "$HERE/dist" -czf "$TARBALL" trapetum-linux
echo "Built $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo "Install with:  tar xzf $(basename "$TARBALL") && sudo ./trapetum-linux/install-linux.sh"
