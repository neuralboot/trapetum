#!/usr/bin/env bash
# Build and package the Trapetum Linux bundle (.tar.gz): serve binary + installer + systemd unit.
# Run on a Linux x86_64 box with the CUDA toolkit and Rust:
#   TRAPETUM_BUILD=2 ./installers/build-linux-bundle.sh 0.1.0-beta.2
# TRAPETUM_BUILD is the build number stamped into the binary. The update check compares it with
# https://cdn.neuralboot.com/dist/latest.json, so every published build needs a new, larger number.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VER="${1:?usage: TRAPETUM_BUILD=<n> build-linux-bundle.sh <version, e.g. 0.1.0-beta.2>}"
: "${TRAPETUM_BUILD:?set TRAPETUM_BUILD to the build number (see the header of this script)}"
export TRAPETUM_BUILD
# a release bundle always carries the portable multi-arch CUDA fatbin (see runtime/build.rs)
if [ -n "${CUDA_ARCH:-}" ]; then echo "note: ignoring CUDA_ARCH=$CUDA_ARCH for a release bundle"; unset CUDA_ARCH; fi
(cd "$ROOT/runtime" && cargo build --release --bin serve)
BIN="$ROOT/runtime/target/release/serve"

STAGE="$HERE/dist/trapetum-linux"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$BIN" "$STAGE/serve"
cp "$HERE/install-linux.sh" "$HERE/uninstall-linux.sh" "$HERE/trapetum.service" "$STAGE/"
echo "$VER" > "$STAGE/VERSION"
chmod +x "$STAGE/"*.sh "$STAGE/serve"

TARBALL="$HERE/dist/trapetum-linux-$VER.tar.gz"
tar -C "$HERE/dist" --owner=0 --group=0 -czf "$TARBALL" trapetum-linux
echo "Built $TARBALL  ($(du -h "$TARBALL" | cut -f1)), build $TRAPETUM_BUILD"
echo "Published as https://cdn.neuralboot.com/dist/trapetum-linux.tar.gz (the file get.neuralboot.com installs)."
echo "Install with:  tar xzf trapetum-linux.tar.gz && sudo ./trapetum-linux/install-linux.sh"
