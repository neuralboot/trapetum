// Trapetum engine, C ABI for native apps (iOS SwiftUI / macOS).
// Link libtrapetum.a. No server, no network: inference runs in-process on the
// Apple GPU (Metal). See src/ffi.rs for the implementation.
#ifndef TRAPETUM_H
#define TRAPETUM_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrapetumSession TrapetumSession;

// Load a model directory holding `model.cbk` + `tokenizer.json`.
// Returns NULL on failure.
TrapetumSession *trapetum_session_new(const char *model_dir);

// Streaming greedy generation. `on_token(piece, user_data)` receives a
// NUL-terminated UTF-8 fragment per token; return false to stop early.
// Returns the number of tokens generated.
int trapetum_generate(TrapetumSession *s,
                      const char *prompt,
                      int max_tokens,
                      bool (*on_token)(const char *piece, void *user_data),
                      void *user_data);

// Free a session from trapetum_session_new.
void trapetum_session_free(TrapetumSession *s);

#ifdef __cplusplus
}
#endif

#endif // TRAPETUM_H
