import Foundation
import SwiftUI

struct ChatMessage: Identifiable, Equatable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var text: String
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input: String = ""
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var modelReady = false
    @Published var statusLine = "Model not loaded"

    private var engine: TrapetumEngine?

    /// The model ships inside the app bundle (or is downloaded to the app's
    /// Documents on first run). Here we look for `<Documents>/models/llama32-1b`.
    private var modelDir: String {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("models/llama32-1b").path
    }

    func loadModel() async {
        guard !modelReady, !isLoading else { return }
        isLoading = true
        statusLine = "Loading model on the Apple GPU…"
        let engine = TrapetumEngine(modelDir: modelDir)
        do {
            try await engine.load()
            self.engine = engine
            modelReady = true
            statusLine = "Ready · runs fully offline on this device"
        } catch {
            statusLine = "Could not load the model. Place model.cbk + tokenizer.json in \(modelDir)."
        }
        isLoading = false
    }

    func send() async {
        let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isGenerating, let engine else { return }
        input = ""
        messages.append(ChatMessage(role: .user, text: prompt))
        var assistant = ChatMessage(role: .assistant, text: "")
        messages.append(assistant)
        let index = messages.count - 1
        isGenerating = true

        // The engine streams tokens on its actor; hop back to the main actor to
        // append each fragment to the visible message.
        let templated = "[INST] \(prompt) [/INST]"
        do {
            try await engine.generate(prompt: templated, maxTokens: 256) { [weak self] piece in
                Task { @MainActor in
                    guard let self, self.messages.indices.contains(index) else { return }
                    self.messages[index].text += piece
                }
                return true
            }
        } catch {
            assistant.text = "(generation error)"
            if messages.indices.contains(index) { messages[index] = assistant }
        }
        isGenerating = false
    }
}
