//! C ABI for embedding the Trapetum engine in a native app (iOS SwiftUI, macOS).
//!
//! No server, no localhost: the app links libtrapetum.a and drives inference
//! in-process. A session owns a Model + Tokenizer; generation streams tokens to
//! a C callback so SwiftUI can render them as they arrive.
//!
//! Lifecycle:
//!   let s = trapetum_session_new(model_dir);   // load .cbk + tokenizer.json
//!   trapetum_generate(s, prompt, max_tokens, cb, user);  // streams token strings
//!   trapetum_session_free(s);
use crate::Model;
use std::ffi::{c_char, c_void, CStr, CString};
use tokenizers::Tokenizer;

pub struct Session {
    model: Model,
    tok: Tokenizer,
    vocab: usize,
}

/// Load a model directory (must contain `model.cbk` and `tokenizer.json`).
/// Returns an opaque session pointer, or null on failure.
///
/// # Safety
/// `model_dir` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn trapetum_session_new(model_dir: *const c_char) -> *mut Session {
    if model_dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let model = match Model::load_cbk(&format!("{dir}/model.cbk"), 2048) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };
    let tok = match Tokenizer::from_file(format!("{dir}/tokenizer.json")) {
        Ok(t) => t,
        Err(_) => return std::ptr::null_mut(),
    };
    let vocab = model.vocab();
    Box::into_raw(Box::new(Session { model, tok, vocab }))
}

/// Streaming greedy generation. `on_token(piece, user_data)` is called once per
/// generated token with a NUL-terminated UTF-8 fragment; return false to stop.
/// Returns the number of tokens generated.
///
/// # Safety
/// `s` must come from `trapetum_session_new`; `prompt` a valid C string.
#[no_mangle]
pub unsafe extern "C" fn trapetum_generate(
    s: *mut Session,
    prompt: *const c_char,
    max_tokens: i32,
    on_token: Option<extern "C" fn(*const c_char, *mut c_void) -> bool>,
    user_data: *mut c_void,
) -> i32 {
    if s.is_null() || prompt.is_null() {
        return 0;
    }
    let sess = &mut *s;
    let prompt = match CStr::from_ptr(prompt).to_str() {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let enc = match sess.tok.encode(prompt, false) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let ids = enc.get_ids();
    if ids.is_empty() {
        return 0;
    }
    let v = sess.vocab;
    // prefill: only the last token's logits are needed, and greedily as an argmax on
    // device (no full-vocab host copy). Earlier tokens just prime the KV cache.
    let mut pos = 0usize;
    let mut next = 0u32;
    let last = ids.len() - 1;
    for (i, &t) in ids.iter().enumerate() {
        if i == last {
            next = sess.model.forward_argmax(t as usize, pos, v);
        } else {
            sess.model.run_forward(t as usize, pos);
        }
        pos += 1;
    }
    // greedy decode, decoding each token to a text piece and streaming it
    let mut produced = Vec::<u32>::new();
    let mut prev_text = String::new();
    let mut n = 0i32;
    while n < max_tokens {
        produced.push(next);
        // incremental detokenization: decode the whole sequence, emit the delta
        let text = sess.tok.decode(&produced, true).unwrap_or_default();
        let piece = text[prev_text.len().min(text.len())..].to_string();
        prev_text = text;
        n += 1;
        if let Some(cb) = on_token {
            if let Ok(cs) = CString::new(piece) {
                if !cb(cs.as_ptr(), user_data) {
                    break;
                }
            }
        }
        next = sess.model.forward_argmax(next as usize, pos, v);
        pos += 1;
    }
    n
}

/// Free a session created by `trapetum_session_new`.
///
/// # Safety
/// `s` must come from `trapetum_session_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn trapetum_session_free(s: *mut Session) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}
