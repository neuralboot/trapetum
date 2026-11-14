#!/usr/bin/env bash
# Build and package the Trapetum macOS bundle (Apple Silicon, Metal): trapetum-macos-arm64.tar.gz
# Run on an Apple Silicon Mac with Xcode, its Metal toolchain
# (xcodebuild -downloadComponent MetalToolchain) and Rust:
#   TRAPETUM_BUILD=2 ./installers/build-macos-bundle.sh 0.1.0-beta.2
# TRAPETUM_BUILD is the build number the update check compares with dist/latest.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VER="${1:?usage: TRAPETUM_BUILD=<n> build-macos-bundle.sh <version, e.g. 0.1.0-beta.2>}"
: "${TRAPETUM_BUILD:?set TRAPETUM_BUILD to the build number (see the header of this script)}"
export TRAPETUM_BUILD
# runtime/build.rs pins the macOS the Metal library targets (13.0, the Metal 3 baseline;
# TRAPETUM_METAL_MACOS_MIN overrides it). Without that pin the library targets the build machine's macOS.
(cd "$ROOT/runtime" && cargo build --release --bin serve --no-default-features --features metal)

STAGE="$HERE/dist/trapetum-macos"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$ROOT/runtime/target/release/serve" "$STAGE/trapetum"
cp "$HERE/macos/install.sh" "$HERE/macos/README.txt" "$STAGE/"
echo "$VER" > "$STAGE/VERSION"
chmod +x "$STAGE/trapetum" "$STAGE/install.sh"

TARBALL="$HERE/dist/trapetum-macos-arm64-$VER.tar.gz"
COPYFILE_DISABLE=1 tar -C "$HERE/dist" --no-mac-metadata --uid 0 --gid 0 --uname root --gname wheel -czf "$TARBALL" trapetum-macos
(cd "$HERE/dist" && shasum -a 256 "trapetum-macos-arm64-$VER.tar.gz" | sed "s/trapetum-macos-arm64-$VER.tar.gz/trapetum-macos-arm64.tar.gz/" > "trapetum-macos-arm64-$VER.tar.gz.sha256")
echo "Built $TARBALL  ($(du -h "$TARBALL" | cut -f1)), build $TRAPETUM_BUILD, Metal library for macOS >= ${TRAPETUM_METAL_MACOS_MIN:-13.0}"
echo "SHA-256 file (publish as dist/trapetum-macos-arm64.tar.gz.sha256, read by get.neuralboot.com/mac): $TARBALL.sha256"
echo "Published as https://cdn.neuralboot.com/dist/trapetum-macos-arm64.tar.gz (the file get.neuralboot.com/mac installs)."
