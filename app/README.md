# Trapetum, iOS/macOS app (skeleton)

A native SwiftUI chat that runs a compressed LLM **fully on-device** via the
Trapetum engine (Rust + Metal). No server, no network during inference.

## Layout
- `Trapetum/Bridge/Trapetum-Bridging-Header.h` — exposes the C engine to Swift.
- `Trapetum/Sources/TrapetumEngine.swift` — actor wrapping the C ABI (streaming).
- `Trapetum/Sources/ChatViewModel.swift` — `@MainActor` chat state, token streaming.
- `Trapetum/Sources/ContentView.swift` — SwiftUI chat UI + `@main` app entry.
- `build-engine.sh` — builds `TrapetumEngine.xcframework` (device + simulator).

## Wire it up in Xcode
1. `./build-engine.sh` → produces `TrapetumEngine.xcframework`.
2. New Xcode iOS App target; add the four Swift files.
3. Drag `TrapetumEngine.xcframework` into the target (Embed & Sign).
4. Build Settings:
   - Objective-C Bridging Header = `Trapetum/Bridge/Trapetum-Bridging-Header.h`
   - Link `Metal.framework`.
5. Ship a model: put `model.cbk` + `tokenizer.json` under
   `<App Documents>/models/llama32-1b/` (bundle it, or download on first run).

## Status
Skeleton only. The engine, C ABI and streaming are validated on macOS (a C driver
runs Llama-3.2-1B through the same symbols). Not yet built into an Xcode project
or run on a device. Next: Xcode project file, first-run model download, thermal /
memory guards, and the 2-bit "capacity mode" model picker.
