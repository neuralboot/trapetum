import SwiftUI

@main
struct TrapetumApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()

    // Brand palette (olive-press identity, matching the website tokens).
    private let ink = Color(red: 0.08, green: 0.09, blue: 0.12)
    private let rust = Color(red: 0.76, green: 0.25, blue: 0.05)
    private let sage = Color(red: 0.64, green: 0.70, blue: 0.34)

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messagesList
            inputBar
        }
        .background(ink.ignoresSafeArea())
        .task { await vm.loadModel() }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Circle().fill(sage).frame(width: 10, height: 10)
            Text("Trapetum").font(.headline).foregroundStyle(.white)
            Spacer()
            Text(vm.statusLine)
                .font(.caption2)
                .foregroundStyle(vm.modelReady ? sage : .secondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    private var messagesList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.messages) { m in
                        MessageBubble(message: m, rust: rust)
                            .id(m.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: vm.messages.last?.text) { _ in
                if let last = vm.messages.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 10) {
            TextField("Ask anything, offline…", text: $vm.input, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(10)
                .background(Color.white.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .foregroundStyle(.white)
                .disabled(!vm.modelReady || vm.isGenerating)

            Button {
                Task { await vm.send() }
            } label: {
                Image(systemName: vm.isGenerating ? "stop.fill" : "arrow.up.circle.fill")
                    .font(.system(size: 30))
                    .foregroundStyle(vm.modelReady ? rust : .gray)
            }
            .disabled(!vm.modelReady || vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(12)
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    let rust: Color

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            Text(message.text.isEmpty ? "…" : message.text)
                .foregroundStyle(.white)
                .padding(12)
                .background(message.role == .user ? rust.opacity(0.85) : Color.white.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 14))
            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }
}
