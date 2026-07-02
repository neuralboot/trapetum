import Foundation

/// Swift wrapper over the Trapetum C engine (libtrapetum.a). Runs entirely
/// on-device: nothing leaves the phone. The engine is not thread-safe, so all
/// calls are serialized on a dedicated actor.
actor TrapetumEngine {
    private var session: OpaquePointer?
    private let modelDir: String

    /// `modelDir` must contain `model.cbk` and `tokenizer.json`.
    init(modelDir: String) {
        self.modelDir = modelDir
    }

    deinit {
        if let s = session { trapetum_session_free(s) }
    }

    enum EngineError: Error { case loadFailed }

    func load() throws {
        guard session == nil else { return }
        guard let s = modelDir.withCString({ trapetum_session_new($0) }) else {
            throw EngineError.loadFailed
        }
        session = s
    }

    /// Streaming greedy generation. `onToken` is called on this actor for each
    /// UTF-8 fragment as it is produced; return false to stop early.
    /// Returns the number of tokens generated.
    @discardableResult
    func generate(prompt: String,
                  maxTokens: Int32 = 256,
                  onToken: @escaping (String) -> Bool) throws -> Int32 {
        if session == nil { try load() }
        guard let s = session else { throw EngineError.loadFailed }

        // Box the Swift closure so the C callback can reach it via user_data.
        final class Sink { let cb: (String) -> Bool; init(_ cb: @escaping (String) -> Bool) { self.cb = cb } }
        let sink = Sink(onToken)
        let user = Unmanaged.passUnretained(sink).toOpaque()

        let cCallback: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Bool = { piece, user in
            guard let piece = piece, let user = user else { return true }
            let sink = Unmanaged<Sink>.fromOpaque(user).takeUnretainedValue()
            return sink.cb(String(cString: piece))
        }

        return prompt.withCString { cPrompt in
            trapetum_generate(s, cPrompt, maxTokens, cCallback, user)
        }
    }
}
