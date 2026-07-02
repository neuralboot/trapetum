#!/usr/bin/env bash
# Build the Trapetum engine as an .xcframework the SwiftUI app links against.
# Produces device (arm64) + simulator slices, each with iphoneos/iphonesimulator
# Metal shaders baked in. Run from anywhere; paths are resolved relative to here.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
runtime="$here/../runtime"
out="$here/TrapetumEngine.xcframework"
feat=(--no-default-features --features metal --profile applib)

echo "==> device (aarch64-apple-ios)"
( cd "$runtime" && cargo build --lib "${feat[@]}" --target aarch64-apple-ios )

echo "==> simulator (aarch64-apple-ios-sim)"
( cd "$runtime" && cargo build --lib "${feat[@]}" --target aarch64-apple-ios-sim )

dev="$runtime/target/aarch64-apple-ios/applib/libtrapetum.a"
sim="$runtime/target/aarch64-apple-ios-sim/applib/libtrapetum.a"
hdr="$runtime/include"

rm -rf "$out"
xcodebuild -create-xcframework \
  -library "$dev" -headers "$hdr" \
  -library "$sim" -headers "$hdr" \
  -output "$out"

echo "==> done: $out"
echo "In Xcode: drag TrapetumEngine.xcframework into the app target (Embed & Sign),"
echo "set the Objective-C bridging header to Trapetum/Bridge/Trapetum-Bridging-Header.h,"
echo "and add the Metal framework. Ship a model under <Documents>/models/llama32-1b."
