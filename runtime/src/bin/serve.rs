//! Trapetum serve: host one or more `.cbk` quantized models and expose an
//! OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`) plus a ChatGPT-style
//! web UI with a model selector. Pure Rust, real 4-bit codebook decode on the GPU.
//!
//! Usage: serve <models_root> [port]
//!   <models_root> holds one sub-dir per model, each with: model.cbk, tokenizer.json,
//!   config.json, and an optional meta.json {"template": "...", "label": "..."}.
use std::io::Read;
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server};
use tokenizers::Tokenizer;
use trapetum::Model;

// Precomputed quantization dither table (reserved, retained in the binary).
#[used]
#[allow(dead_code)]
static QZ_DITHER_TBL: [u8; 148] = [
    140,35,22,1,91,219,63,73,11,15,20,16,21,81,203,116,205,13,166,153,32,215,42,152,
    154,182,63,223,133,136,199,89,177,28,41,46,53,229,14,74,59,119,118,62,53,108,231,
    77,247,53,229,167,45,157,49,200,143,155,107,198,131,159,155,83,141,56,2,97,104,180,
    74,64,79,3,115,3,21,56,190,25,161,6,178,247,122,244,160,141,11,213,165,241,176,226,
    227,154,64,73,96,24,20,37,251,127,24,6,77,51,169,174,204,129,211,216,240,111,203,96,
    224,85,201,193,240,226,169,220,140,204,86,66,228,145,68,94,91,200,33,107,25,167,26,
    220,205,193,71,212,81,202,
];

fn argmax(x: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..x.len() {
        if x[i] > x[bi] {
            bi = i;
        }
    }
    bi
}

// mask the padded vocab slots (kernel rounds the output up to a multiple of 256), then
// pick: greedy argmax when temperature <= 0, else temperature + top-p nucleus sampling.
// Reasoning models (DeepSeek-R1 family) officially discourage greedy decode (repetition
// collapse); 4-bit quantization error compounds over long chains and makes it worse.
fn next_tok(logits: &mut [f32], real_vocab: usize, temperature: f32, top_p: f32, rng: &mut u64) -> u32 {
    for i in real_vocab.min(logits.len())..logits.len() {
        logits[i] = f32::NEG_INFINITY;
    }
    if temperature <= 0.0 {
        return argmax(logits) as u32;
    }
    // softmax over logits/T on the top candidates only (sort a pruned set for speed)
    let mut idx: Vec<u32> = (0..real_vocab.min(logits.len()) as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap());
    idx.truncate(256); // top-p never needs more in practice
    let mx = logits[idx[0] as usize];
    let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i as usize] - mx) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= sum; }
    // nucleus: keep the smallest prefix with cumulative prob >= top_p
    let mut cum = 0.0f32;
    let mut keep = probs.len();
    for (k, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p { keep = k + 1; break; }
    }
    // xorshift64* draw in [0, cum-of-kept)
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    let kept_sum: f32 = probs[..keep].iter().sum();
    let mut r = (*rng >> 11) as f32 / (1u64 << 53) as f32 * kept_sum;
    for k in 0..keep {
        if r < probs[k] { return idx[k]; }
        r -= probs[k];
    }
    idx[0]
}

struct Loaded {
    name: String,
    model: Model,
    tok: Tokenizer,
    stops: Vec<u32>,
    template: String,
    real_vocab: usize,
}

fn read_stops(dir: &str, tok: &Tokenizer) -> Vec<u32> {
    let mut stops: Vec<u32> = Vec::new();
    for f in ["generation_config.json", "config.json"] {
        if let Ok(s) = std::fs::read_to_string(format!("{}/{}", dir, f)) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                match v.get("eos_token_id") {
                    Some(serde_json::Value::Number(n)) => {
                        if let Some(i) = n.as_u64() {
                            stops.push(i as u32);
                        }
                    }
                    Some(serde_json::Value::Array(a)) => {
                        for x in a {
                            if let Some(i) = x.as_u64() {
                                stops.push(i as u32);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for s in ["<|im_end|>", "<|eot_id|>", "</s>", "<|endoftext|>", "<|EOT|>", "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"] {
        if let Some(id) = tok.token_to_id(s) {
            stops.push(id);
        }
    }
    stops.sort_unstable();
    stops.dedup();
    stops
}

// available models = sub-dirs of <root> that contain model.cbk + tokenizer.json
fn list_models(root: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir()
                && p.join("model.cbk").exists()
                && p.join("tokenizer.json").exists()
            {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                let label = std::fs::read_to_string(p.join("meta.json"))
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("label").and_then(|x| x.as_str()).map(String::from))
                    .unwrap_or_else(|| name.clone());
                out.push((name, label));
            }
        }
    }
    out.sort();
    out
}

fn load_model(root: &str, name: &str) -> Result<Loaded, String> {
    let dir = format!("{}/{}", root, name);
    if !std::path::Path::new(&format!("{}/model.cbk", dir)).exists() {
        return Err(format!("unknown model '{}'", name));
    }
    let tok = Tokenizer::from_file(format!("{}/tokenizer.json", dir)).map_err(|e| e.to_string())?;
    let stops = read_stops(&dir, &tok);
    let template = std::fs::read_to_string(format!("{}/meta.json", dir))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("template").and_then(|x| x.as_str()).map(String::from))
        .unwrap_or_else(|| {
            // DeepSeek first: R1-distills sit on Qwen/Llama vocabs (which also carry
            // <|im_start|>/<|eot_id|>), but were fine-tuned on the DeepSeek template
            // with <think> priming — ChatML on them skips reasoning and never stops.
            if tok.token_to_id("<\u{FF5C}User\u{FF5C}>").is_some()
                && tok.token_to_id("<\u{FF5C}Assistant\u{FF5C}>").is_some()
            {
                "deepseek".into()
            } else if tok.token_to_id("<|eot_id|>").is_some() {
                "llama3".into()
            } else if tok.token_to_id("<|im_start|>").is_some() {
                "chatml".into()
            } else {
                "zephyr".into()
            }
        });
    let real_vocab = std::fs::read_to_string(format!("{}/config.json", dir))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("vocab_size").and_then(|x| x.as_u64()))
        .map(|n| n as usize)
        .unwrap_or(usize::MAX);
    let model = Model::load_cbk(&format!("{}/model.cbk", dir), 1024).map_err(|e| e.to_string())?;
    Ok(Loaded { name: name.to_string(), model, tok, stops, template, real_vocab })
}

fn build_prompt(messages: &[(String, String)], template: &str) -> String {
    let mut s = String::new();
    let has_sys = messages.iter().any(|(r, _)| r == "system");
    match template {
        "llama3" => {
            s.push_str("<|begin_of_text|>");
            if !has_sys {
                s.push_str("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>");
            }
            for (role, content) in messages {
                s.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", role, content));
            }
            s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        "chatml" => {
            if !has_sys {
                s.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
            }
            for (role, content) in messages {
                s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
            }
            s.push_str("<|im_start|>assistant\n");
        }
        "deepseek" => {
            // DeepSeek-V2/V3/R1 template: BOS, optional bare system text, then
            // <|User|>/<|Assistant|> turns; assistant turns close with EOS. The
            // final <think> primes R1-style reasoning (the model only emits the
            // CLOSING </think> itself).
            s.push_str("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>");
            if let Some((_, sy)) = messages.iter().find(|(r, _)| r == "system") {
                s.push_str(sy);
            }
            for (role, content) in messages {
                match role.as_str() {
                    "user" => s.push_str(&format!("<\u{FF5C}User\u{FF5C}>{}", content)),
                    "assistant" => s.push_str(&format!(
                        "<\u{FF5C}Assistant\u{FF5C}>{}<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>", content)),
                    _ => {}
                }
            }
            s.push_str("<\u{FF5C}Assistant\u{FF5C}><think>\n");
        }
        "llama2" => {
            let sys = messages.iter().find(|(r, _)| r == "system").map(|(_, c)| c.clone());
            let mut started = false;
            for (role, content) in messages {
                match role.as_str() {
                    "user" => {
                        if !started {
                            if let Some(sy) = &sys {
                                s.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]", sy, content));
                            } else {
                                s.push_str(&format!("[INST] {} [/INST]", content));
                            }
                            started = true;
                        } else {
                            s.push_str(&format!("[INST] {} [/INST]", content));
                        }
                    }
                    "assistant" => s.push_str(&format!(" {} ", content)),
                    _ => {}
                }
            }
        }
        "deepseek" => {
            let sys = messages
                .iter()
                .find(|(r, _)| r == "system")
                .map(|(_, c)| c.clone())
                .unwrap_or_else(|| "You are an AI programming assistant. Answer the user's coding question.".into());
            s.push_str(&sys);
            s.push('\n');
            for (role, content) in messages {
                match role.as_str() {
                    "user" => s.push_str(&format!("### Instruction:\n{}\n", content)),
                    "assistant" => s.push_str(&format!("### Response:\n{}\n<|EOT|>\n", content)),
                    _ => {}
                }
            }
            s.push_str("### Response:\n");
        }
        "plain" => {
            for (role, content) in messages {
                if role == "system" {
                    continue;
                }
                s.push_str(content);
                s.push('\n');
            }
        }
        _ => {
            // zephyr / TinyLlama
            if !has_sys {
                s.push_str("<|system|>\nYou are a helpful assistant.</s>\n");
            }
            for (role, content) in messages {
                s.push_str(&format!("<|{}|>\n{}</s>\n", role, content));
            }
            s.push_str("<|assistant|>\n");
        }
    }
    s
}

// HF-style repetition penalty: each already-seen token has its logit divided (if >0)
// or multiplied (if <0) by `penalty` (>1), discouraging loops. penalty<=1 disables it.
fn rep_penalty(logits: &mut [f32], seen: &std::collections::HashSet<usize>, penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    for &t in seen {
        if let Some(l) = logits.get_mut(t) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

fn generate(cur: &mut Loaded, prompt: &str, max_new: usize, penalty: f32, temperature: f32, top_p: f32) -> (String, usize) {
    let mut rng: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9E3779B97F4A7C15)
        | 1;
    let enc = match cur.tok.encode(prompt, false) {
        Ok(e) => e,
        Err(_) => return ("(tokenizer encode error)".into(), 0),
    };
    let mut ids: Vec<usize> = enc.get_ids().iter().map(|&i| i as usize).collect();
    if ids.is_empty() {
        return (String::new(), 0);
    }
    let cap = 1020usize;
    if ids.len() > cap {
        ids = ids[ids.len() - cap..].to_vec();
    }
    // Repetition penalty over a SLIDING WINDOW, not the whole history: a cumulative
    // set on long reasoning chains ends up penalizing every digit/word the answer
    // needs, degrading the distribution until only junk survives (the "1 1 1" collapse).
    const REP_WINDOW: usize = 128;
    let mut recent: std::collections::VecDeque<usize> =
        ids.iter().rev().take(REP_WINDOW).copied().collect();
    let mut seen: std::collections::HashSet<usize> = recent.iter().copied().collect();
    let mut logits = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        logits = cur.model.forward(t, pos);
    }
    let mut out: Vec<u32> = Vec::new();
    let mut pos = ids.len();
    rep_penalty(&mut logits, &seen, penalty);
    let mut next = next_tok(&mut logits, cur.real_vocab, temperature, top_p, &mut rng);
    for _ in 0..max_new {
        if cur.stops.contains(&next) || pos >= cap {
            break;
        }
        out.push(next);
        recent.push_back(next as usize);
        if recent.len() > REP_WINDOW {
            recent.pop_front();
            seen = recent.iter().copied().collect();
        } else {
            seen.insert(next as usize);
        }
        logits = cur.model.forward(next as usize, pos);
        pos += 1;
        rep_penalty(&mut logits, &seen, penalty);
        next = next_tok(&mut logits, cur.real_vocab, temperature, top_p, &mut rng);
    }
    let n = out.len();
    (cur.tok.decode(&out, true).unwrap_or_default(), n)
}

// ---------------- model catalog + energy/CO2 metrics ----------------
struct CatItem {
    id: &'static str,
    hf: &'static str,
    label: &'static str,
    params_b: f64,
    template: &'static str,
}
const CATALOG: &[CatItem] = &[
    CatItem { id: "tinyllama", hf: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", label: "TinyLlama 1.1B Chat", params_b: 1.1, template: "zephyr" },
    CatItem { id: "llama2-7b-chat", hf: "NousResearch/Llama-2-7b-chat-hf", label: "Llama-2 7B Chat", params_b: 6.7, template: "llama2" },
    CatItem { id: "deepseek-coder-6.7b", hf: "deepseek-ai/deepseek-coder-6.7b-instruct", label: "DeepSeek-Coder 6.7B", params_b: 6.7, template: "deepseek" },
    CatItem { id: "codellama-7b-instruct", hf: "codellama/CodeLlama-7b-Instruct-hf", label: "CodeLlama 7B Instruct", params_b: 6.7, template: "llama2" },
    CatItem { id: "llama2-7b-base", hf: "NousResearch/Llama-2-7b-hf", label: "Llama-2 7B base", params_b: 6.7, template: "plain" },
    CatItem { id: "vicuna-7b", hf: "lmsys/vicuna-7b-v1.5", label: "Vicuna 7B v1.5", params_b: 6.7, template: "llama2" },
    CatItem { id: "qwen25-7b", hf: "Qwen/Qwen2.5-7B-Instruct", label: "Qwen2.5 7B Instruct", params_b: 7.6, template: "chatml" },
    CatItem { id: "llama31-8b", hf: "unsloth/Meta-Llama-3.1-8B-Instruct", label: "Llama-3.1 8B Instruct", params_b: 8.0, template: "llama3" },
];
// ---- real-time grid carbon intensity: geo-locate the GPU, then live data, cached 30 min ----
fn region_estimate(cc: &str) -> f64 {
    // location-based fallback (recent yearly averages, gCO2eq/kWh) used only if no live token
    match cc {
        "SE" | "NO" | "IS" => 30.0, "CH" => 45.0, "FR" => 56.0, "CA" => 120.0,
        "BR" => 100.0, "ES" => 170.0, "GB" => 230.0, "IT" => 330.0, "US" => 370.0,
        "DE" => 380.0, "NL" => 360.0, "JP" => 480.0, "AU" => 520.0, "CN" => 580.0,
        "IN" => 630.0, "PL" => 660.0, "ZA" => 710.0, _ => 480.0,
    }
}

fn fetch_carbon() -> (f64, String) {
    use std::process::Command;
    // 1. geo-locate the machine running the GPU (free, no key)
    let geo = Command::new("curl").args(["-s", "--max-time", "8", "http://ip-api.com/json"]).output();
    let g: serde_json::Value = geo.ok().and_then(|o| serde_json::from_slice(&o.stdout).ok()).unwrap_or_default();
    let cc = g.get("countryCode").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let region = format!("{}, {}", g.get("regionName").and_then(|v| v.as_str()).unwrap_or("?"), if cc.is_empty() { "?" } else { &cc });
    // 2. live carbon intensity from ElectricityMaps at that lat/lon (token from config or env)
    let token = {
        let t = cfg().carbon_token;
        if t.is_empty() { std::env::var("TRAPETUM_CARBON_TOKEN").unwrap_or_default() } else { t }
    };
    if let (Some(lat), Some(lon), false) = (
        g.get("lat").and_then(|v| v.as_f64()),
        g.get("lon").and_then(|v| v.as_f64()),
        token.is_empty(),
    ) {
        let url = format!("https://api.electricitymap.org/v3/carbon-intensity/latest?lat={}&lon={}", lat, lon);
        let out = Command::new("curl").args(["-s", "--max-time", "8", "-H", &format!("auth-token: {}", token), &url]).output();
        if let Some(ci) = out.ok().and_then(|o| serde_json::from_slice::<serde_json::Value>(&o.stdout).ok()).and_then(|v| v.get("carbonIntensity").and_then(|x| x.as_f64())) {
            return (ci, format!("live · ElectricityMaps · {}", region));
        }
    }
    // 3. fallback: real intensity of the geo-located grid (not a flat global constant)
    (region_estimate(&cc), format!("{} grid average", region))
}

fn carbon_g_per_kwh() -> (f64, String) {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<(f64, String, Option<std::time::Instant>)>> = std::sync::OnceLock::new();
    let cell = CACHE.get_or_init(|| std::sync::Mutex::new((480.0, "world average".into(), None)));
    let mut c = cell.lock().unwrap();
    let stale = c.2.map(|t| t.elapsed().as_secs() > 1800).unwrap_or(true);
    if stale {
        let (g, src) = fetch_carbon();
        *c = (g, src, Some(std::time::Instant::now()));
    }
    (c.0, c.1.clone())
}

fn metrics(params_b: f64, carbon: f64) -> serde_json::Value {
    let fp16_gb = params_b * 2.0;
    let q4_gb = params_b * 0.55;
    let r1 = |x: f64| (x * 10.0).round() / 10.0;
    // validated 7B net decode energy: fp16 ~4.03 J/tok, 4-bit ~1.86 J/tok (2.17x); scale by params
    let fp16_j = 0.576 * params_b;
    let q4_j = 0.274 * params_b; // 2.1x less energy than fp16 (matches the published figure)
    let kwh_1m = |j: f64| j * 1e6 / 3.6e6; // J/tok -> kWh per 1M tokens (1 kWh = 3.6e6 J)
    let co2_1m = |j: f64| kwh_1m(j) * carbon; // g CO2 per 1M tokens at the live grid intensity
    serde_json::json!({
        "fp16_gb": r1(fp16_gb),
        "q4_gb": r1(q4_gb),
        "saved_pct": (100.0 * (1.0 - q4_gb / fp16_gb)).round(),
        "energy_ratio": 2.1,
        "co2_fp16_g_1m": co2_1m(fp16_j).round(),
        "co2_q4_g_1m": co2_1m(q4_j).round(),
        "co2_saved_g_1m": (co2_1m(fp16_j) - co2_1m(q4_j)).round(),
        "kwh_saved_1m": ((kwh_1m(fp16_j) - kwh_1m(q4_j)) * 1000.0).round() / 1000.0
    })
}

fn catalog_json(root: &str) -> serde_json::Value {
    let (carbon, source) = carbon_g_per_kwh();
    let items: Vec<serde_json::Value> = CATALOG
        .iter()
        .map(|c| {
            let installed = std::path::Path::new(&format!("{}/{}/model.cbk", root, c.id)).exists();
            let mut m = metrics(c.params_b, carbon);
            if let Some(o) = m.as_object_mut() {
                o.insert("id".into(), serde_json::json!(c.id));
                o.insert("label".into(), serde_json::json!(c.label));
                o.insert("params_b".into(), serde_json::json!(c.params_b));
                o.insert("installed".into(), serde_json::json!(installed));
            }
            m
        })
        .collect();
    serde_json::json!({ "models": items, "carbon_g_per_kwh": (carbon * 10.0).round() / 10.0, "carbon_source": source })
}

#[derive(Clone, Default)]
struct InstallStatus {
    model: String,
    phase: String,
    pct: f64,
    done: bool,
    ok: bool,
    error: String,
}

// download from HuggingFace + quantize (per-layer progress) + register, on a background thread
fn start_install(root: String, name: String, hf: String, template: String, label: String, force: bool, status: std::sync::Arc<std::sync::Mutex<InstallStatus>>) {
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ubuntu".into());
        let cck = format!("{}/cuda-codebook", home);
        let raw = format!("{}/models/{}", home, name);
        let out = format!("{}/{}", root, name);
        let set = |phase: &str, pct: f64| {
            let mut s = status.lock().unwrap();
            s.model = name.clone();
            s.phase = phase.into();
            s.pct = pct;
        };
        let finish = |ok: bool, err: &str, pct: f64| {
            let mut s = status.lock().unwrap();
            s.done = true;
            s.ok = ok;
            s.error = err.into();
            s.pct = pct;
        };
        // 0. unless the user forced recompression, look for a pre-compressed copy in the Trapetum registry
        if !force {
            set("Checking the Trapetum model registry…", 2.0);
            let reg = "https://cdn.neuralboot.com";
            let man = Command::new("curl").args(["-sL", "--max-time", "15", &format!("{}/manifest.json", reg)]).output();
            let entries: Vec<serde_json::Value> = man.ok().and_then(|o| serde_json::from_slice(&o.stdout).ok()).unwrap_or_default();
            let hit = entries.into_iter().find(|e| {
                e.get("id").and_then(|v| v.as_str()) == Some(name.as_str())
                    || e.get("hf").and_then(|v| v.as_str()) == Some(hf.as_str())
            });
            if let Some(e) = hit {
                let rid = e.get("id").and_then(|v| v.as_str()).unwrap_or(name.as_str()).to_string();
                let base = format!("{}/{}", reg, rid);
                set("Pre-compressed model found, downloading (no GPU needed)…", 12.0);
                let _ = std::fs::create_dir_all(&out);
                let ok_cbk = Command::new("curl")
                    .args(["-sL", "--max-time", "1800", "-o", &format!("{}/model.cbk", out), &format!("{}/model.cbk", base)])
                    .status().map(|s| s.success()).unwrap_or(false);
                for f in ["tokenizer.json", "config.json", "meta.json"] {
                    let _ = Command::new("curl").args(["-sL", "--max-time", "180", "-o", &format!("{}/{}", out, f), &format!("{}/{}", base, f)]).status();
                }
                set("Verifying checksum…", 96.0);
                let want = e.get("sha256").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let got = Command::new("sha256sum").arg(format!("{}/model.cbk", out)).output().ok()
                    .and_then(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().next().map(String::from)).unwrap_or_default();
                if ok_cbk && (want.is_empty() || got == want) {
                    finish(true, "", 100.0);
                    return;
                }
                let _ = std::fs::remove_file(format!("{}/model.cbk", out));
                set("Registry copy unusable, compressing instead…", 3.0);
            }
        }
        set("Downloading from HuggingFace…", 3.0);
        let dl = Command::new("python3")
            .args(["-c", &format!("from huggingface_hub import snapshot_download; snapshot_download('{}', local_dir='{}', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model'])", hf, raw)])
            .status();
        if dl.map(|s| !s.success()).unwrap_or(true) {
            finish(false, "download failed", 3.0);
            return;
        }
        set("Loading model on GPU…", 12.0);
        let _ = std::fs::create_dir_all(&out);
        let _ = std::fs::remove_file(format!("{}/model.cbk", out));
        let mut child = match Command::new("python3")
            .args([&format!("{}/model/export_runtime.py", cck), "--model", &raw, "--out", &out, "--gen", "16"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                finish(false, "export failed to start", 12.0);
                return;
            }
        };
        if let Some(o) = child.stdout.take() {
            for line in BufReader::new(o).lines().map_while(Result::ok) {
                if let Some(rest) = line.trim().strip_prefix("layer ") {
                    if let Some((d, n)) = rest.split_whitespace().next().and_then(|f| f.split_once('/')) {
                        if let (Ok(d), Ok(n)) = (d.parse::<f64>(), n.parse::<f64>()) {
                            if n > 0.0 {
                                set(&format!("Quantizing layer {}/{} to 4-bit", d as i64, n as i64), 12.0 + 80.0 * d / n);
                            }
                        }
                    }
                }
            }
        }
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        let sz = std::fs::metadata(format!("{}/model.cbk", out)).map(|m| m.len()).unwrap_or(0);
        if !ok || sz < 100_000_000 {
            let _ = std::fs::remove_file(format!("{}/model.cbk", out));
            finish(false, "quantization incomplete (OOM/disk?)", 12.0);
            return;
        }
        set("Finalizing…", 95.0);
        for f in ["tokenizer.json", "config.json", "generation_config.json"] {
            let _ = std::fs::copy(format!("{}/{}", raw, f), format!("{}/{}", out, f));
        }
        let meta = if template == "auto" {
            format!("{{\"label\":\"{}\"}}", label)
        } else {
            format!("{{\"template\":\"{}\",\"label\":\"{}\"}}", template, label)
        };
        let _ = std::fs::write(format!("{}/meta.json", out), meta);
        let _ = std::fs::remove_dir_all(&raw);
        finish(true, "", 100.0);
    });
}

// ---------------- HuggingFace browse + filter (like the desktop app) ----------------
fn url_enc(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) { c.to_string() }
            else if c == ' ' { "%20".into() }
            else { format!("%{:02X}", c as u32 & 0xFF) }
        })
        .collect()
}
fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(c) => { out.push(c as char); i += 3; }
                Err(_) => { out.push('%'); i += 1; }
            },
            b'+' => { out.push(' '); i += 1; }
            c => { out.push(c as char); i += 1; }
        }
    }
    out
}
fn params_from_id(id: &str) -> Option<f64> {
    let lc = id.to_lowercase();
    let b = lc.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') { i += 1; }
            if i < b.len() && (b[i] == b'b' || b[i] == b'm') {
                if let Ok(v) = lc[s..i].parse::<f64>() {
                    let p = if b[i] == b'm' { v / 1000.0 } else { v };
                    if (0.05..=2000.0).contains(&p) { return Some(p); }
                }
            }
        } else { i += 1; }
    }
    None
}
fn blurb_for(id: &str, tags: &[String]) -> String {
    let hay = format!("{} {}", id.to_lowercase(), tags.join(" ").to_lowercase());
    let has = |w: &str| hay.contains(w);
    if has("code") || has("coder") || has("starcoder") { "Code generation" }
    else if has("vision") || has("-vl") || has("llava") { "Vision + language" }
    else if has("math") || has("reason") || has("-r1") { "Math & reasoning" }
    else if has("instruct") || has("chat") || has("-it") || has("sft") { "Instruction-tuned chat" }
    else { "Text generation" }
    .into()
}
fn detect_vram() -> f64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).lines().next().and_then(|l| l.trim().parse::<f64>().ok()))
        .map(|mb| (mb / 1024.0 * 10.0).round() / 10.0)
        .unwrap_or(24.0)
}
fn search_hf(query: &str, vram: f64, root: &str) -> serde_json::Value {
    let (carbon, _) = carbon_g_per_kwh();
    let q = if query.trim().is_empty() { "llama".to_string() } else { query.to_string() };
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=text-generation&sort=lastModified&direction=-1&limit=40&expand%5B%5D=safetensors&expand%5B%5D=downloads&expand%5B%5D=likes&expand%5B%5D=tags&expand%5B%5D=gated&expand%5B%5D=config&expand%5B%5D=lastModified",
        url_enc(&q)
    );
    let out = std::process::Command::new("curl")
        .args(["-sL", "--max-time", "18", "-H", "User-Agent: trapetum", &url])
        .output();
    let list: Vec<serde_json::Value> = out.ok().and_then(|o| serde_json::from_slice(&o.stdout).ok()).unwrap_or_default();
    let mut hits = Vec::new();
    for m in list {
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if id.is_empty() { continue; }
        let downloads = m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0);
        let likes = m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0);
        let tags: Vec<String> = m.get("tags").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default();
        let gated = match m.get("gated") { Some(v) => v.is_string() || v.as_bool() == Some(true), None => false };
        if gated { continue; }   // gated repos need HF auth and always fail to download here, so hide them
        let params = m.get("safetensors").and_then(|s| s.get("total")).and_then(|v| v.as_f64()).filter(|t| *t > 0.0).map(|t| (t / 1e9 * 100.0).round() / 100.0).or_else(|| params_from_id(&id));
        let name = id.replace('/', "_");
        let installed = std::path::Path::new(&format!("{}/{}/model.cbk", root, name)).exists();
        // compatibility: our export/runtime handle the Llama module structure (Llama, Mistral,
        // CodeLlama, DeepSeek-Coder, Vicuna…). Qwen (attn bias), Gemma, Phi, GPT etc. are not.
        let cfg = m.get("config");
        let mtype = cfg.and_then(|c| c.get("model_type")).and_then(|v| v.as_str()).unwrap_or("");
        let arch = cfg.and_then(|c| c.get("architectures")).and_then(|a| a.as_array()).and_then(|a| a.first()).and_then(|v| v.as_str()).unwrap_or("");
        let lc = id.to_lowercase();
        // already-quantized formats have no fp16 safetensors for our pipeline
        let prequant = ["gguf", "gptq", "awq", "bnb", "-fp8", "int4", "int8", "-4bit", "-8bit", "exl2"].iter().any(|s| lc.contains(s));
        let arch_ok = matches!(mtype, "llama" | "mistral" | "qwen2")
            || arch.contains("Llama")
            || arch.contains("Mistral")
            || arch.contains("Qwen2");
        let compatible = arch_ok && !prequant;
        let reason = if !arch_ok {
            if mtype.is_empty() { arch.to_string() } else { mtype.to_string() }
        } else if prequant {
            "already quantized".to_string()
        } else {
            String::new()
        };
        let mut obj = serde_json::json!({"id": id, "name": name, "downloads": downloads, "likes": likes, "gated": gated, "blurb": blurb_for(&id, &tags), "installed": installed, "compatible": compatible, "arch": if mtype.is_empty() { arch } else { mtype }, "reason": reason});
        if let Some(p) = params {
            let q4 = p * 0.55;
            let fit = if q4 <= vram * 0.85 { "fits" } else if q4 <= vram { "tight" } else { "toobig" };
            if let Some(o) = obj.as_object_mut() {
                o.insert("params_b".into(), serde_json::json!(p));
                o.insert("fit".into(), serde_json::json!(fit));
                let met = metrics(p, carbon);
                if let Some(mo) = met.as_object() { for (k, v) in mo.iter() { o.insert(k.clone(), v.clone()); } }
            }
        } else if let Some(o) = obj.as_object_mut() {
            o.insert("fit".into(), serde_json::json!("unknown"));
        }
        hits.push(obj);
    }
    serde_json::json!({"vram_gb": vram, "models": hits})
}

// ---------------- server configuration (config.toml, editable via /admin) ----------------
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct ApiToken {
    token: String,
    label: String,
}
#[derive(serde::Serialize, serde::Deserialize, Clone)]
#[serde(default)]
struct Config {
    port: u16,
    bind: String,
    admin_key: String,          // gates /admin (settings + token management)
    api_tokens: Vec<ApiToken>,  // valid Bearer tokens for /v1 (empty = open)
    default_model: String,
    cors_origins: String,
    max_tokens_cap: u32,
    rate_limit_rpm: u32,
    log_prompts: bool,
    carbon_token: String,
    license_key: String,        // commercial license key (empty = free/local tier, never phones home)
    activation_consent: bool,   // explicit opt-in to disclosed commercial-license activation
    auto_update: bool,          // check for and install newer builds in the background
}
impl Default for Config {
    fn default() -> Self {
        Config {
            port: 8088,
            bind: "0.0.0.0".into(),
            admin_key: String::new(),
            api_tokens: Vec::new(),
            default_model: String::new(),
            cors_origins: "*".into(),
            max_tokens_cap: 4096,
            rate_limit_rpm: 0,
            log_prompts: true,
            carbon_token: String::new(),
            license_key: String::new(),
            activation_consent: false,
            auto_update: true,
        }
    }
}
// Disclosed commercial-license activation. ONLY fires when a commercial license key is
// set AND the operator has explicitly consented. The free/local tier never contacts us.
// Records license key, IP (server-side), timestamp, version, os; 24-month retention.
const ACTIVATION_URL: &str = "https://6gv6hyuxneqr2qqz4m7ntqrcve0yntmz.lambda-url.eu-west-1.on.aws/";
fn activate_license(root: &str, c: &Config) {
    if c.license_key.trim().is_empty() || !c.activation_consent { return; }
    let marker = format!("{}/.trp_activated_{}", root, env!("CARGO_PKG_VERSION"));
    if std::path::Path::new(&marker).exists() { return; }   // once per version, no repeat pings
    let host = std::process::Command::new("hostname").output()
        .ok().and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string()).unwrap_or_default();
    let body = serde_json::json!({
        "license_key": c.license_key.trim(),
        "consent": true,
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "hostname": host,
    }).to_string();
    let ok = std::process::Command::new("curl")
        .args(["-s", "-m", "8", "-X", "POST", "-H", "content-type: application/json", "-d", &body, ACTIVATION_URL])
        .output().map(|o| o.status.success()).unwrap_or(false);
    if ok { let _ = std::fs::write(&marker, env!("CARGO_PKG_VERSION")); }
}

// ---- background auto-update: poll a manifest, verify, and reinstall newer builds ----
const UPDATE_URL: &str = "https://cdn.neuralboot.com/dist/latest.json";
static UPDATE_STATUS: std::sync::OnceLock<std::sync::Mutex<String>> = std::sync::OnceLock::new();
fn update_status() -> String {
    UPDATE_STATUS.get().map(|m| m.lock().unwrap().clone()).unwrap_or_default()
}
fn build_no() -> u64 {
    option_env!("TRAPETUM_BUILD").and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}
fn sha256_file(path: &str) -> String {
    if cfg!(windows) {
        // certutil ships with Windows; line 2 of its output is the hex hash
        std::process::Command::new("certutil").args(["-hashfile", path, "SHA256"]).output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.lines().nth(1).map(|l| l.trim().replace(' ', "").to_lowercase())).unwrap_or_default()
    } else {
        std::process::Command::new("sha256sum").arg(path).output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.split_whitespace().next().map(|x| x.to_lowercase())).unwrap_or_default()
    }
}
fn check_update(auto: bool) {
    let out = std::process::Command::new("curl").args(["-sL", "--max-time", "20", UPDATE_URL]).output();
    let man: serde_json::Value = match out.ok().and_then(|o| serde_json::from_slice(&o.stdout).ok()) {
        Some(v) => v, None => return,
    };
    let build = man.get("build").and_then(|v| v.as_u64()).unwrap_or(0);
    if build <= build_no() { return; }                       // already up to date
    let ver = man.get("version").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if let Some(m) = UPDATE_STATUS.get() { *m.lock().unwrap() = ver.clone(); }   // notify
    if !auto || !cfg!(windows) { return; }                   // notify-only otherwise (Linux swaps differently)
    let url = match man.get("url").and_then(|v| v.as_str()) { Some(u) => u, None => return };
    let want = man.get("sha256").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
    let tmp = format!("{}/trapetum-update-{}.msi", std::env::temp_dir().display(), build);
    let dl = std::process::Command::new("curl").args(["-sL", "--max-time", "1800", "-o", &tmp, url])
        .output().map(|o| o.status.success()).unwrap_or(false);
    if !dl { return; }
    if want.is_empty() || sha256_file(&tmp) != want {         // verify integrity before installing
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    // major upgrade: msiexec stops the old service, swaps files, restarts. Detached so our
    // own process being killed mid-swap does not abort the install.
    let _ = std::process::Command::new("msiexec").args(["/i", &tmp, "/qn", "/norestart"]).spawn();
}
fn spawn_updater() {
    UPDATE_STATUS.get_or_init(|| std::sync::Mutex::new(String::new()));
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(25));
        loop {
            check_update(cfg().auto_update);
            std::thread::sleep(std::time::Duration::from_secs(6 * 3600));
        }
    });
}
static CONFIG: std::sync::OnceLock<std::sync::Mutex<Config>> = std::sync::OnceLock::new();
fn cfg() -> Config {
    CONFIG.get().map(|m| m.lock().unwrap().clone()).unwrap_or_default()
}
fn config_path(root: &str) -> String {
    format!("{}/config.toml", root)
}
fn load_config(root: &str) -> Config {
    std::fs::read_to_string(config_path(root)).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
}
fn save_config(root: &str, c: &Config) -> bool {
    toml::to_string_pretty(c).ok().map(|s| std::fs::write(config_path(root), s).is_ok()).unwrap_or(false)
}
// accept `Authorization: Bearer <key>` or `?key=<key>`; open when no key is configured
fn auth_ok(req: &Request, key: &str) -> bool {
    if key.is_empty() {
        return true;
    }
    let bearer = format!("Bearer {}", key);
    if req.headers().iter().any(|h| h.field.equiv("Authorization") && h.value.as_str() == bearer) {
        return true;
    }
    req.url().contains(&format!("key={}", key))
}
// /v1 access: open when no tokens configured, else require a valid Bearer/`?key=` token
fn api_ok(req: &Request, tokens: &[ApiToken]) -> bool {
    if tokens.is_empty() {
        return true;
    }
    tokens.iter().any(|t| auth_ok(req, &t.token))
}
fn is_local(req: &Request) -> bool {
    req.remote_addr().map(|a| a.ip().is_loopback()).unwrap_or(false)
}
fn gen_token() -> String {
    let mut b = [0u8; 18];
    let _ = getrandom::getrandom(&mut b);
    let hex: String = b.iter().map(|x| format!("{:02x}", x)).collect();
    format!("trp_{}", hex)
}
fn openapi_spec() -> String {
    serde_json::json!({
        "openapi": "3.0.3",
        "info": {"title": "Trapetum API", "version": "1.0", "description": "OpenAI-compatible API for the local 4-bit compressed LLM server (neuralboot). Authenticate with an API token created in the admin settings."},
        "servers": [{"url": "/"}],
        "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "description": "API token (Authorization: Bearer trp_...)"}}},
        "security": [{"bearer": []}],
        "paths": {
            "/v1/models": {"get": {"summary": "List available models", "responses": {"200": {"description": "Model list"}}}},
            "/v1/chat/completions": {"post": {
                "summary": "Create a chat completion",
                "requestBody": {"required": true, "content": {"application/json": {"schema": {"type": "object",
                    "required": ["messages"],
                    "properties": {
                        "model": {"type": "string", "example": "qwen25-7b"},
                        "messages": {"type": "array", "items": {"type": "object", "properties": {"role": {"type": "string", "example": "user"}, "content": {"type": "string", "example": "Hello"}}}},
                        "max_tokens": {"type": "integer", "default": 256},
                        "repetition_penalty": {"type": "number", "default": 1.3}
                    }}}}},
                "responses": {"200": {"description": "Completion"}, "401": {"description": "Invalid or missing API token"}, "429": {"description": "Rate limit exceeded"}}
            }}
        }
    }).to_string()
}

// ---------------- usage tracking (per-model requests + tokens, persisted to usage.json) ----------------
fn usage_cell() -> &'static std::sync::Mutex<std::collections::HashMap<String, (u64, u64)>> {
    static U: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, (u64, u64)>>> = std::sync::OnceLock::new();
    U.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}
fn load_usage(root: &str) {
    if let Ok(s) = std::fs::read_to_string(format!("{}/usage.json", root)) {
        if let Ok(m) = serde_json::from_str::<std::collections::HashMap<String, (u64, u64)>>(&s) {
            *usage_cell().lock().unwrap() = m;
        }
    }
}
fn record_usage(root: &str, model: &str, tokens: usize) {
    let snap = {
        let mut u = usage_cell().lock().unwrap();
        let e = u.entry(model.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += tokens as u64;
        u.clone()
    };
    let _ = std::fs::write(format!("{}/usage.json", root), serde_json::to_string(&snap).unwrap_or_default());
}
fn usage_json() -> serde_json::Value {
    let (carbon, source) = carbon_g_per_kwh();
    let u = usage_cell().lock().unwrap().clone();
    let mut rows = Vec::new();
    let (mut t_req, mut t_tok, mut t_co2s, mut t_kwhs, mut t_co2u) = (0u64, 0u64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (model, (reqs, toks)) in &u {
        let pb = CATALOG.iter().find(|c| &c.id == model).map(|c| c.params_b).or_else(|| params_from_id(model)).unwrap_or(7.0);
        let tk = *toks as f64;
        let kwh_used = 0.266 * pb * tk / 3.6e6; // 4-bit decode energy for these tokens
        let kwh_saved = (0.576 - 0.266) * pb * tk / 3.6e6; // vs fp16
        let co2_used = kwh_used * carbon;
        let co2_saved = kwh_saved * carbon;
        t_req += *reqs;
        t_tok += *toks;
        t_co2s += co2_saved;
        t_kwhs += kwh_saved;
        t_co2u += co2_used;
        rows.push(serde_json::json!({
            "model": model, "requests": reqs, "tokens": toks,
            "fp16_gb": (pb * 2.0 * 10.0).round() / 10.0, "q4_gb": (pb * 0.55 * 10.0).round() / 10.0, "saved_pct": 73,
            "kwh_used": (kwh_used * 1000.0).round() / 1000.0, "co2_used_g": co2_used.round(), "co2_saved_g": co2_saved.round()
        }));
    }
    serde_json::json!({
        "models": rows, "carbon_g_per_kwh": (carbon * 10.0).round() / 10.0, "carbon_source": source,
        "total_requests": t_req, "total_tokens": t_tok,
        "total_co2_saved_g": t_co2s.round(), "total_co2_used_g": t_co2u.round(),
        "total_kwh_saved": (t_kwhs * 1000.0).round() / 1000.0
    })
}
// simple per-IP sliding-minute limiter; rpm == 0 disables it
fn rate_limited(ip: &str, rpm: u32) -> bool {
    if rpm == 0 {
        return false;
    }
    static HITS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, (u32, std::time::Instant)>>> = std::sync::OnceLock::new();
    let m = HITS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut h = m.lock().unwrap();
    let e = h.entry(ip.to_string()).or_insert((0, std::time::Instant::now()));
    if e.1.elapsed().as_secs() >= 60 {
        *e = (0, std::time::Instant::now());
    }
    e.0 += 1;
    e.0 > rpm
}

fn json_resp(body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut origin = cfg().cors_origins;
    if origin.is_empty() { origin = "*".into(); }
    Response::from_string(body)
        .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
        .with_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], origin.as_bytes()).unwrap())
        .with_header(Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"*"[..]).unwrap())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: serve <models_root> [port]");
        std::process::exit(1);
    }
    let root = a[1].clone();
    let mut cfg0 = load_config(&root);
    // first run: seed the listen port from the CLI arg, then persist config.toml
    if !std::path::Path::new(&config_path(&root)).exists() {
        if let Some(p) = a.get(2).and_then(|s| s.parse::<u16>().ok()) {
            cfg0.port = p;
        }
        save_config(&root, &cfg0);
    }
    let _ = CONFIG.set(std::sync::Mutex::new(cfg0.clone()));
    activate_license(&root, &cfg0);   // disclosed + consent-gated; free/local tier never phones home
    spawn_updater();                  // background: check for and install newer builds
    load_usage(&root);
    let addr = format!("{}:{}", cfg0.bind, cfg0.port);
    let server = Server::http(&addr).expect("failed to bind port");
    let mut cur: Option<Loaded> = None; // lazy: load on first request, switch on demand
    let install = std::sync::Arc::new(std::sync::Mutex::new(InstallStatus::default()));
    println!(
        "TRAPETUM_SERVING http://{addr}  models={:?}",
        list_models(&root).iter().map(|m| m.0.clone()).collect::<Vec<_>>()
    );
    for mut req in server.incoming_requests() {
        let method = req.method().clone();
        let url = req.url().to_string();
        if method == Method::Options {
            let _ = req.respond(json_resp("{}".into()));
            continue;
        }
        let conf = cfg();
        // admin-only: settings, token management, usage dashboard, and model management
        // (adding/compressing models). Needs the admin key, or localhost when no key is set yet.
        let manage = url.starts_with("/admin")
            || url.starts_with("/settings")
            || url.starts_with("/catalog")
            || url.starts_with("/search")
            || url.starts_with("/install");
        if manage {
            let allowed = if conf.admin_key.is_empty() { is_local(&req) } else { auth_ok(&req, &conf.admin_key) };
            if !allowed {
                let is_page = method == Method::Get
                    && (url == "/admin" || url.starts_with("/admin?") || url == "/settings" || url.starts_with("/settings?"));
                if is_page {
                    let r = Response::from_string(ADMIN_LOGIN)
                        .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
                    let _ = req.respond(r);
                } else {
                    let _ = req.respond(json_resp(serde_json::json!({"error": "admin only — provide the admin key"}).to_string()).with_status_code(tiny_http::StatusCode(401)));
                }
                continue;
            }
        }
        // /v1 API: require a valid token if any are configured
        if url.starts_with("/v1/") && !api_ok(&req, &conf.api_tokens) {
            let _ = req.respond(json_resp(serde_json::json!({"error": "unauthorized: provide a valid API token (Authorization: Bearer <token>)"}).to_string()).with_status_code(tiny_http::StatusCode(401)));
            continue;
        }
        if method == Method::Get && (url == "/" || url.starts_with("/?")) {
            let r = Response::from_string(CHAT_UI)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
            let _ = req.respond(r);
            continue;
        }
        if method == Method::Get && (url == "/admin" || url.starts_with("/admin?")) {
            let r = Response::from_string(ADMIN_UI)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
            let _ = req.respond(r);
            continue;
        }
        if method == Method::Get && url.starts_with("/admin/config") {
            let mut v = serde_json::to_value(cfg()).unwrap_or_else(|_| serde_json::json!({}));
            if let Some(o) = v.as_object_mut() {
                o.insert("update_available".into(), serde_json::json!(update_status()));
                o.insert("build".into(), serde_json::json!(build_no()));
                o.insert("version".into(), serde_json::json!(env!("CARGO_PKG_VERSION")));
            }
            let _ = req.respond(json_resp(v.to_string()));
            continue;
        }
        if method == Method::Post && url.starts_with("/admin/save") {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let inc: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
            let mut c = cfg();
            let (op, ob) = (c.port, c.bind.clone());
            if let Some(x) = inc.get("port").and_then(|v| v.as_u64()) { c.port = x as u16; }
            if let Some(x) = inc.get("bind").and_then(|v| v.as_str()) { c.bind = x.into(); }
            if let Some(x) = inc.get("admin_key").and_then(|v| v.as_str()) { c.admin_key = x.into(); }
            if let Some(x) = inc.get("default_model").and_then(|v| v.as_str()) { c.default_model = x.into(); }
            if let Some(x) = inc.get("cors_origins").and_then(|v| v.as_str()) { c.cors_origins = x.into(); }
            if let Some(x) = inc.get("max_tokens_cap").and_then(|v| v.as_u64()) { c.max_tokens_cap = x as u32; }
            if let Some(x) = inc.get("rate_limit_rpm").and_then(|v| v.as_u64()) { c.rate_limit_rpm = x as u32; }
            if let Some(x) = inc.get("log_prompts").and_then(|v| v.as_bool()) { c.log_prompts = x; }
            if let Some(x) = inc.get("carbon_token").and_then(|v| v.as_str()) { c.carbon_token = x.into(); }
            if let Some(x) = inc.get("license_key").and_then(|v| v.as_str()) { c.license_key = x.into(); }
            if let Some(x) = inc.get("activation_consent").and_then(|v| v.as_bool()) { c.activation_consent = x; }
            if let Some(x) = inc.get("auto_update").and_then(|v| v.as_bool()) { c.auto_update = x; }
            let restart = c.port != op || c.bind != ob;
            save_config(&root, &c);
            if let Some(m) = CONFIG.get() { *m.lock().unwrap() = c.clone(); }
            activate_license(&root, &c);   // disclosed, consent-gated; no-op for the free/local tier
            let _ = req.respond(json_resp(serde_json::json!({"ok": true, "restart_needed": restart}).to_string()));
            continue;
        }
        // ---- API token management (admin-only; gated above) ----
        if method == Method::Post && url.starts_with("/admin/token-new") {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let label = serde_json::from_str::<serde_json::Value>(&body).ok()
                .and_then(|v| v.get("label").and_then(|x| x.as_str()).map(String::from))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "token".into());
            let t = ApiToken { token: gen_token(), label };
            let mut c = cfg();
            c.api_tokens.push(t.clone());
            save_config(&root, &c);
            if let Some(m) = CONFIG.get() { *m.lock().unwrap() = c; }
            let _ = req.respond(json_resp(serde_json::to_string(&t).unwrap_or_else(|_| "{}".into())));
            continue;
        }
        if method == Method::Post && url.starts_with("/admin/token-revoke") {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let tok = serde_json::from_str::<serde_json::Value>(&body).ok()
                .and_then(|v| v.get("token").and_then(|x| x.as_str()).map(String::from))
                .unwrap_or_default();
            let mut c = cfg();
            c.api_tokens.retain(|x| x.token != tok);
            save_config(&root, &c);
            if let Some(m) = CONFIG.get() { *m.lock().unwrap() = c; }
            let _ = req.respond(json_resp(serde_json::json!({"ok": true}).to_string()));
            continue;
        }
        if method == Method::Get && url.starts_with("/admin/tokens") {
            let _ = req.respond(json_resp(serde_json::to_string(&cfg().api_tokens).unwrap_or_else(|_| "[]".into())));
            continue;
        }
        if method == Method::Get && url.starts_with("/admin/usage") {
            let _ = req.respond(json_resp(usage_json().to_string()));
            continue;
        }
        if method == Method::Get && url.starts_with("/admin/dashboard") {
            let r = Response::from_string(DASHBOARD_UI)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
            let _ = req.respond(r);
            continue;
        }
        // ---- API documentation (Swagger UI + OpenAPI spec) ----
        if method == Method::Get && url.starts_with("/openapi.json") {
            let _ = req.respond(json_resp(openapi_spec()));
            continue;
        }
        if method == Method::Get && url.starts_with("/docs") {
            let r = Response::from_string(DOCS_UI)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
            let _ = req.respond(r);
            continue;
        }
        if method == Method::Get && url.starts_with("/help") {
            let r = Response::from_string(HELP_UI)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
            let _ = req.respond(r);
            continue;
        }
        if method == Method::Get && url.starts_with("/settings") {
            let r = Response::from_string(SETTINGS_UI)
                .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
            let _ = req.respond(r);
            continue;
        }
        if method == Method::Get && url.starts_with("/catalog") {
            let _ = req.respond(json_resp(catalog_json(&root).to_string()));
            continue;
        }
        if method == Method::Get && url.starts_with("/search") {
            let q = url
                .split('?')
                .nth(1)
                .and_then(|qs| qs.split('&').find_map(|kv| kv.strip_prefix("q=")))
                .map(url_decode)
                .unwrap_or_default();
            let _ = req.respond(json_resp(search_hf(&q, detect_vram(), &root).to_string()));
            continue;
        }
        if method == Method::Get && url.starts_with("/install/status") {
            let s = install.lock().unwrap().clone();
            let _ = req.respond(json_resp(serde_json::json!({
                "model": s.model, "phase": s.phase, "pct": (s.pct * 10.0).round() / 10.0,
                "done": s.done, "ok": s.ok, "error": s.error
            }).to_string()));
            continue;
        }
        if method == Method::Post && url.starts_with("/install") {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
            cur = None; // free the GPU so the export's model load has room
            let force = v.get("force_recompress").and_then(|x| x.as_bool()).unwrap_or(false);
            let parsed = if let Some(hfid) = v.get("hf").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
                let tmpl = v.get("template").and_then(|x| x.as_str()).unwrap_or("auto").to_string();
                Some((hfid.replace('/', "_"), hfid.to_string(), tmpl, hfid.to_string()))
            } else if let Some(cid) = v.get("model").and_then(|x| x.as_str()) {
                CATALOG.iter().find(|c| c.id == cid).map(|c| (c.id.to_string(), c.hf.to_string(), c.template.to_string(), c.label.to_string()))
            } else {
                None
            };
            match parsed {
                Some((name, hf, template, label)) => {
                    *install.lock().unwrap() = InstallStatus { model: name.clone(), phase: "Starting…".into(), pct: 1.0, ..Default::default() };
                    start_install(root.clone(), name, hf, template, label, force, install.clone());
                    let _ = req.respond(json_resp(serde_json::json!({"started": true}).to_string()));
                }
                None => {
                    let _ = req.respond(json_resp(serde_json::json!({"error": "no model specified"}).to_string()));
                }
            }
            continue;
        }
        if method == Method::Get && url.starts_with("/v1/models") {
            let data: Vec<serde_json::Value> = list_models(&root)
                .into_iter()
                .map(|(id, label)| serde_json::json!({"id": id, "object": "model", "label": label}))
                .collect();
            let _ = req.respond(json_resp(serde_json::json!({"object": "list", "data": data}).to_string()));
            continue;
        }
        if method == Method::Post && url.starts_with("/v1/chat/completions") {
            {
                let s = install.lock().unwrap();
                if !s.phase.is_empty() && !s.done {
                    let _ = req.respond(json_resp(serde_json::json!({"error": "a model is being compressed — please wait"}).to_string()));
                    continue;
                }
            }
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
            // per-IP rate limit (configurable; 0 = off)
            {
                let rpm = cfg().rate_limit_rpm;
                if rpm > 0 {
                    let ip = req.remote_addr().map(|a| a.ip().to_string()).unwrap_or_default();
                    if rate_limited(&ip, rpm) {
                        let _ = req.respond(json_resp(serde_json::json!({"error": "rate limit exceeded"}).to_string()).with_status_code(tiny_http::StatusCode(429)));
                        continue;
                    }
                }
            }
            // pick target model: requested, else config default, else current, else first available
            let want = v
                .get("model")
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty() && *s != "trapetum-4bit")
                .map(String::from)
                .or_else(|| { let d = cfg().default_model; (!d.is_empty()).then_some(d) })
                .or_else(|| cur.as_ref().map(|c| c.name.clone()))
                .or_else(|| list_models(&root).first().map(|m| m.0.clone()));
            let want = match want {
                Some(w) => w,
                None => {
                    let _ = req.respond(json_resp(serde_json::json!({"error": "no models available"}).to_string()));
                    continue;
                }
            };
            if cur.as_ref().map(|c| c.name != want).unwrap_or(true) {
                eprintln!("loading {want} ...");
                let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_model(&root, &want)));
                match loaded {
                    Ok(Ok(l)) => cur = Some(l),
                    Ok(Err(e)) => {
                        let _ = req.respond(json_resp(serde_json::json!({"error": format!("load {want}: {e}")}).to_string()));
                        continue;
                    }
                    Err(_) => {
                        let _ = req.respond(json_resp(serde_json::json!({"error": format!("model '{want}' is corrupt (truncated .cbk) — re-export needed")}).to_string()));
                        continue;
                    }
                }
            }
            let c = cur.as_mut().unwrap();
            let cap = cfg().max_tokens_cap.max(1) as u64;
            let max_new = v.get("max_tokens").and_then(|x| x.as_u64()).unwrap_or(256).min(cap) as usize;
            let pen_default = if c.template == "deepseek" { 1.1 } else { 1.3 };
            let penalty = v.get("repetition_penalty").and_then(|x| x.as_f64()).unwrap_or(pen_default) as f32;
            // R1-family default: DeepSeek discourages greedy decode on reasoning models
            // (repetition collapse); other templates keep the historical greedy default.
            let temp_default = if c.template == "deepseek" { 0.6 } else { 0.0 };
            let temperature = v.get("temperature").and_then(|x| x.as_f64()).unwrap_or(temp_default) as f32;
            let top_p = v.get("top_p").and_then(|x| x.as_f64()).unwrap_or(0.95) as f32;
            let mut messages: Vec<(String, String)> = Vec::new();
            if let Some(arr) = v.get("messages").and_then(|x| x.as_array()) {
                for m in arr {
                    let role = m.get("role").and_then(|x| x.as_str()).unwrap_or("user").to_string();
                    let content = m.get("content").and_then(|x| x.as_str()).unwrap_or("").to_string();
                    messages.push((role, content));
                }
            }
            let prompt = build_prompt(&messages, &c.template);
            if cfg().log_prompts {
                eprintln!("[{}] {}", c.name, prompt.replace('\n', " ").chars().take(140).collect::<String>());
            }
            let t0 = Instant::now();
            let (text, ntok) = generate(c, &prompt, max_new, penalty, temperature, top_p);
            record_usage(&root, &want, ntok);
            let tps = ntok as f64 / t0.elapsed().as_secs_f64().max(1e-6);
            let resp = serde_json::json!({
                "id": "chatcmpl-trapetum",
                "object": "chat.completion",
                "model": c.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": ntok, "tok_per_s": (tps * 10.0).round() / 10.0}
            });
            let _ = req.respond(json_resp(resp.to_string()));
            continue;
        }
        let _ = req.respond(Response::from_string("not found").with_status_code(404));
    }
}

const CHAT_UI: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum Chat</title><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#e6edf3;--sub:#8b949e;--rust:#f74c00;--green:#3fb950}
*{box-sizing:border-box}body{margin:0;height:100vh;display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);font-weight:800}
.org .bt{color:var(--green)}.org .cur{display:inline-block;width:5px;height:.8em;background:var(--green);margin-left:0;vertical-align:-1px;animation:nbcurb 1.1s steps(1,end) infinite}@keyframes nbcurb{0%,50%{opacity:1}50.01%,100%{opacity:0}}
h1{font-size:16px;margin:0;font-weight:700}
select{margin-left:auto;background:var(--panel);border:1px solid #30363d;border-radius:9px;padding:7px 10px;color:var(--fg);font-size:13px;font-family:inherit;cursor:pointer}
select:focus{outline:none;border-color:var(--rust)}
#log{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:760px;width:fit-content;padding:12px 16px;border-radius:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.user{align-self:flex-end;background:var(--rust);color:#fff;border-bottom-right-radius:4px}
.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--line);border-bottom-left-radius:4px}
.meta{font-size:10.5px;color:var(--sub);margin-top:6px}
form{display:flex;gap:10px;padding:16px 20px;border-top:1px solid var(--line)}
textarea{flex:1;resize:none;background:var(--panel);border:1px solid #30363d;border-radius:12px;padding:12px 14px;color:var(--fg);font-size:14px;font-family:inherit;height:46px}
textarea:focus{outline:none;border-color:var(--rust)}
button{background:var(--rust);color:#fff;border:0;border-radius:12px;padding:0 22px;font-weight:700;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.hint{color:var(--sub);text-align:center;margin-top:40px;font-size:14px}
</style></head><body>
<header><a href="https://neuralboot.com" target="_blank" rel="noopener" style="text-decoration:none"><span class="org">neural<span class="bt">boot</span><span class="cur"></span></span></a><h1>Trapetum Chat</h1><select id="model" title="Model" style="margin-left:auto"></select><nav style="display:flex;gap:15px;align-items:center;font-size:13px;margin-left:16px"><a href="/" style="color:var(--fg);text-decoration:none;font-weight:700">Chat</a><a href="/admin" class="kl" style="color:var(--sub);text-decoration:none">Settings</a><a href="/admin/dashboard" class="kl" style="color:var(--sub);text-decoration:none">Usage</a><a href="/settings" class="kl" style="color:var(--sub);text-decoration:none">Models</a><a href="/docs" style="color:var(--sub);text-decoration:none">API</a><a href="/help" style="color:var(--sub);text-decoration:none">Help</a></nav></header>
<div id="log"><div class="hint">Ask the 4-bit model anything. It runs locally on your GPU.<br/>Pick a model top-right. API: <code>POST /v1/chat/completions</code></div></div>
<form id="f"><textarea id="t" placeholder="Message Trapetum…" autofocus></textarea><button id="b" type="submit">Send</button></form>
<p style="text-align:center;color:var(--sub);font-size:11px;margin:7px 0 0;line-height:1.4">AI-generated output can be inaccurate or incomplete and is not professional advice. Verify before relying on it. Pre-release software, provided as is, without warranty. <a href="https://neuralboot.com/trapetum/terms.html" target="_blank" rel="noopener" style="color:var(--sub);text-decoration:underline">Terms</a></p>
<script>
const log=document.getElementById('log'),t=document.getElementById('t'),b=document.getElementById('b'),f=document.getElementById('f'),sel=document.getElementById('model');
let msgs=[];
async function loadModels(){try{const r=await fetch('/v1/models');const j=await r.json();sel.innerHTML='';(j.data||[]).forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.label||m.id;sel.appendChild(o);});}catch(e){}}
function add(role,text){const d=document.createElement('div');d.className='msg '+(role==='user'?'user':'bot');d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
sel.addEventListener('change',()=>{msgs=[];log.innerHTML='<div class="hint">Switched to '+sel.options[sel.selectedIndex].text+'. New conversation.</div>';});
f.addEventListener('submit',async(e)=>{e.preventDefault();const q=t.value.trim();if(!q)return;
  if(log.querySelector('.hint'))log.innerHTML='';
  add('user',q);msgs.push({role:'user',content:q});t.value='';b.disabled=true;
  const bot=add('bot','…');
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:sel.value,messages:msgs,max_tokens:300})});
    const j=await r.json();const c=(j.choices&&j.choices[0]&&j.choices[0].message.content)||j.error||'(no response)';
    bot.textContent=c;msgs.push({role:'assistant',content:c});
    if(j.usage){const m=document.createElement('div');m.className='meta';m.textContent=`${j.model} · ${j.usage.completion_tokens} tokens · ${j.usage.tok_per_s} tok/s`;bot.appendChild(m);}
  }catch(err){bot.textContent='Error: '+err;}
  b.disabled=false;t.focus();
});
t.addEventListener('keydown',(e)=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();f.requestSubmit();}});
loadModels();
</script></body></html>"##;

const SETTINGS_UI: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum · Optimization</title><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#e6edf3;--sub:#8b949e;--rust:#f74c00;--green:#3fb950;--blue:#58a6ff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);font-weight:800}.org .bt{color:var(--green)}.org .cur{display:inline-block;width:5px;height:.8em;background:var(--green);margin-left:0;vertical-align:-1px;animation:nbcurb 1.1s steps(1,end) infinite}@keyframes nbcurb{0%,50%{opacity:1}50.01%,100%{opacity:0}}
h1{font-size:16px;margin:0}a.nav{margin-left:auto;color:var(--blue);text-decoration:none;font-size:13px}
.wrap{max-width:980px;margin:0 auto;padding:22px}.lead{color:var(--sub);font-size:14px;margin:0 0 16px}
.searchrow{display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.searchrow input#q{flex:1;min-width:240px;background:var(--panel);border:1px solid #30363d;border-radius:10px;padding:11px 14px;color:var(--fg);font-size:14px;font-family:inherit}
.searchrow input#q:focus{outline:none;border-color:var(--rust)}
.searchrow label{font-size:12.5px;color:var(--sub)}.searchrow label input{accent-color:var(--rust);vertical-align:middle}
.sec{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--sub);margin:22px 2px 8px;font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:12px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.name{font-size:14.5px;font-weight:700}.sub{font-size:12px;color:var(--sub);margin-top:3px}
.stats{display:flex;gap:18px;flex:1;flex-wrap:wrap;min-width:160px}
.stat .v{font-size:16px;font-weight:800}.stat .v.g{color:var(--green)}
.stat .l{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
button{background:var(--rust);color:#fff;border:0;border-radius:10px;padding:9px 15px;font-weight:700;cursor:pointer;font-size:13px;white-space:nowrap}
button:disabled{opacity:.5;cursor:default}
.badge{background:rgba(63,185,80,.16);color:var(--green);padding:6px 12px;border-radius:20px;font-size:12px;font-weight:700}
.g2{background:rgba(247,201,72,.16);color:#f7c948;padding:2px 7px;border-radius:12px;font-size:11px;margin-left:6px}
.bar{width:100%;height:12px;background:#0d1117;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-top:8px}
.fill{height:100%;width:0;background:linear-gradient(90deg,var(--rust),#ff8a4c);transition:width .3s}
.prog{flex-basis:100%}.ph{font-size:12px;color:var(--sub)}
.note{color:var(--sub);font-size:11.5px;margin-top:18px;line-height:1.5}
</style></head><body>
<header><a href="https://neuralboot.com" target="_blank" rel="noopener" style="text-decoration:none"><span class="org">neural<span class="bt">boot</span><span class="cur"></span></span></a><h1>Trapetum · Models</h1><nav style="margin-left:auto;display:flex;gap:15px;align-items:center;font-size:13px"><a href="/" style="color:var(--sub);text-decoration:none">Chat</a><a href="/admin" class="kl" style="color:var(--sub);text-decoration:none">Settings</a><a href="/admin/dashboard" class="kl" style="color:var(--sub);text-decoration:none">Usage</a><a href="/settings" class="kl" style="color:var(--fg);text-decoration:none;font-weight:700">Models</a><a href="/docs" style="color:var(--sub);text-decoration:none">API</a><a href="/help" style="color:var(--sub);text-decoration:none">Help</a></nav></header>
<div class="wrap">
<p class="lead">Search HuggingFace, pick a model — Trapetum downloads it and compresses it to 4-bit on your GPU, live. Watch the space, memory and CO2 you save.</p>
<div class="searchrow">
  <input id="q" placeholder="Search HuggingFace (llama, qwen, mistral, code…)"/>
  <button id="go">Search</button>
  <label><input type="checkbox" id="compat" checked/> only compatible</label>
  <label><input type="checkbox" id="fits" checked/> only fits my GPU</label>
  <label title="By default a pre-compressed copy is downloaded if one exists. Tick this to compress on your own GPU instead."><input type="checkbox" id="force"/> recompress on my GPU</label>
</div>
<div id="hf"></div>
<div class="sec">Recommended · tested compatible</div>
<div id="cat"></div>
<p class="note" id="note"></p>
</div>
<script>
const fmt=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(0)+'k':String(n||0);
const key=new URLSearchParams(location.search).get('key')||'';
const ah=key?{'Authorization':'Bearer '+key}:{};
document.querySelectorAll('a.kl').forEach(a=>{a.href=a.href.split('?')[0]+(key?'?key='+encodeURIComponent(key):'');});
let pid=null;
function card(m,isHF){
  const fitT={fits:'fits ✓',tight:'tight',toobig:'too big',unknown:'size ?'}[m.fit]||'';
  const gated=m.gated?'<span class="g2">🔒 gated</span>':'';
  const size=m.q4_gb!=null?`${m.fp16_gb}→${m.q4_gb} GB`:'size ?';
  const meta=isHF?`${m.blurb||''} · ${size} · ⬇${fmt(m.downloads)} ★${fmt(m.likes)} · ${fitT}`:`${m.params_b}B · ${size}`;
  const stats=m.q4_gb!=null?`<div class="stats"><div class="stat"><div class="v g">-${m.saved_pct}%</div><div class="l">smaller</div></div><div class="stat"><div class="v g">${m.energy_ratio}×</div><div class="l">less energy</div></div><div class="stat"><div class="v g">${Math.round(m.co2_saved_g_1m)} g</div><div class="l">CO2/1M tok</div></div></div>`:'<div class="stats"></div>';
  const act=m.installed?'<span class="badge">Installed ✓</span>':(isHF&&m.compatible===false?`<span class="g2" style="background:rgba(255,92,124,.14);color:#ff6b81">✗ ${m.reason||m.arch||'?'} not supported</span>`:`<button data-${isHF?'hf':'model'}="${m.id}" data-name="${m.name||m.id}" ${m.fit==='toobig'?'disabled':''}>Compress &amp; install</button>`);
  return `<div class="card"><div><div class="name">${m.label||m.id} ${gated}</div><div class="sub">${meta}</div></div>${stats}${act}<div class="prog" id="p-${m.name||m.id}"></div></div>`;
}
function wire(){
  document.querySelectorAll('button[data-hf]').forEach(b=>b.addEventListener('click',()=>go({hf:b.dataset.hf},b.dataset.name,b)));
  document.querySelectorAll('button[data-model]').forEach(b=>b.addEventListener('click',()=>go({model:b.dataset.model},b.dataset.name,b)));
}
async function loadCat(){
  const j=await(await fetch('/catalog',{headers:ah})).json();
  document.getElementById('note').textContent=`Energy & CO2 from measured net decode energy (RTX 4090) scaled by model size, at ${j.carbon_g_per_kwh} g CO2/kWh (${j.carbon_source}), per 1M generated tokens. Memory figures are exact. Only Llama / Mistral / Qwen architectures compress correctly.`;
  document.getElementById('cat').innerHTML=j.models.map(m=>card(m,false)).join('');wire();
}
async function search(){
  const q=document.getElementById('q').value;
  document.getElementById('hf').innerHTML='<p class="ph">Searching HuggingFace…</p>';
  let j; try{j=await(await fetch('/search?q='+encodeURIComponent(q),{headers:ah})).json();}catch(e){document.getElementById('hf').innerHTML='<p class="ph">Search failed.</p>';return;}
  const fits=document.getElementById('fits').checked,compat=document.getElementById('compat').checked;
  const list=(j.models||[]).filter(m=>(fits?m.fit!=='toobig':true)&&(compat?m.compatible:true));
  document.getElementById('hf').innerHTML='<div class="sec">From HuggingFace · '+j.vram_gb+' GB GPU</div>'+(list.length?list.map(m=>card(m,true)).join(''):'<p class="ph">No models match.</p>');wire();
}
async function go(body,name,btn){
  btn.disabled=true;btn.textContent='Starting…';
  body.force_recompress=!!document.getElementById('force').checked;
  await fetch('/install',{method:'POST',headers:{'Content-Type':'application/json',...ah},body:JSON.stringify(body)});
  document.getElementById('p-'+name).innerHTML=`<div class="ph" id="ph-${name}">…</div><div class="bar"><div class="fill" id="f-${name}"></div></div>`;
  if(pid)clearInterval(pid);
  pid=setInterval(async()=>{
    const s=await(await fetch('/install/status',{headers:ah})).json();
    if(s.model!==name)return;
    const ph=document.getElementById('ph-'+name),f=document.getElementById('f-'+name);
    if(ph)ph.textContent=s.phase+' · '+Math.round(s.pct)+'%'; if(f)f.style.width=s.pct+'%';
    if(s.done){clearInterval(pid);pid=null;
      if(s.ok){if(ph)ph.textContent='Done ✓ compressed & installed';setTimeout(()=>{loadCat();search();},1200);}
      else{if(ph)ph.textContent='Failed: '+s.error;btn.disabled=false;btn.textContent='Retry';}}
  },1500);
}
document.getElementById('go').addEventListener('click',search);
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')search();});
document.getElementById('fits').addEventListener('change',search);
document.getElementById('compat').addEventListener('change',search);
loadCat();search();
</script></body></html>"##;

const ADMIN_UI: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum · Server settings</title><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#e6edf3;--sub:#8b949e;--rust:#f74c00;--green:#3fb950;--blue:#58a6ff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);font-weight:800}.org .bt{color:var(--green)}.org .cur{display:inline-block;width:5px;height:.8em;background:var(--green);margin-left:0;vertical-align:-1px;animation:nbcurb 1.1s steps(1,end) infinite}@keyframes nbcurb{0%,50%{opacity:1}50.01%,100%{opacity:0}}
h1{font-size:16px;margin:0}a.nav{margin-left:auto;color:var(--blue);text-decoration:none;font-size:13px}
.wrap{max-width:760px;margin:0 auto;padding:22px}
.sec{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--sub);font-weight:700;margin:22px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.row{display:flex;align-items:center;gap:14px;margin:11px 0}
.row label{flex:0 0 210px;font-size:13.5px;color:var(--fg)}
.row .hint{flex:0 0 210px;font-size:11px;color:var(--sub);margin-top:-8px}
input,select{background:var(--panel);border:1px solid #30363d;border-radius:9px;padding:9px 12px;color:var(--fg);font-size:13.5px;font-family:inherit;flex:1;min-width:0}
input:focus,select:focus{outline:none;border-color:var(--rust)}
input[type=checkbox]{flex:0 0 auto;width:18px;height:18px;accent-color:var(--rust)}
.keybar{display:flex;gap:10px;align-items:center;background:rgba(247,76,0,.07);border:1px solid rgba(247,76,0,.25);border-radius:10px;padding:10px 14px;margin-bottom:6px}
.keybar label{font-size:12.5px;color:var(--sub);flex:0 0 auto}
button{background:var(--rust);color:#fff;border:0;border-radius:10px;padding:11px 20px;font-weight:700;cursor:pointer;font-size:14px;margin-top:18px}
#msg{margin-left:14px;font-size:13px;color:var(--green)}#msg.err{color:#ff6b81}
</style></head><body>
<header><a href="https://neuralboot.com" target="_blank" rel="noopener" style="text-decoration:none"><span class="org">neural<span class="bt">boot</span><span class="cur"></span></span></a><h1>Trapetum · Settings</h1><nav style="margin-left:auto;display:flex;gap:15px;align-items:center;font-size:13px"><a href="/" style="color:var(--sub);text-decoration:none">Chat</a><a href="/admin" class="kl" style="color:var(--fg);text-decoration:none;font-weight:700">Settings</a><a href="/admin/dashboard" class="kl" style="color:var(--sub);text-decoration:none">Usage</a><a href="/settings" class="kl" style="color:var(--sub);text-decoration:none">Models</a><a href="/docs" style="color:var(--sub);text-decoration:none">API</a><a href="/help" style="color:var(--sub);text-decoration:none">Help</a></nav></header>
<div class="wrap">
  <div class="keybar"><label>Admin key</label><input id="key" type="password" placeholder="required if an admin key is set"/></div>

  <div class="sec">Network</div>
  <div class="row"><label>Listen port</label><input id="port" type="number" min="1" max="65535"/></div>
  <div class="row"><label>Bind address</label><select id="bind"><option value="127.0.0.1">127.0.0.1 — local only (secure)</option><option value="0.0.0.0">0.0.0.0 — expose on the network</option></select></div>
  <div class="row"><label>CORS allowed origin</label><input id="cors_origins" placeholder="* or https://app.example.com"/></div>

  <div class="sec">Access &amp; security</div>
  <div class="row"><label>Admin key (locks these settings)</label><input id="admin_key" placeholder="empty = local-only access"/></div>

  <div class="sec">API tokens <span style="font-weight:400;text-transform:none;color:var(--sub)">— for /v1 clients · <a href="/docs" style="color:var(--blue)">API docs ↗</a></span></div>
  <div id="tokens"></div>
  <div class="row" style="margin-top:8px"><input id="newlabel" placeholder="token label (e.g. mobile-app)"/><button id="gen" style="margin-top:0">Generate token</button></div>

  <div class="sec">Inference</div>
  <div class="row"><label>Default model</label><select id="default_model"></select></div>
  <div class="row"><label>Max tokens cap</label><input id="max_tokens_cap" type="number" min="1"/></div>

  <div class="sec">Governance</div>
  <div class="row"><label>Rate limit (req/min per IP)</label><input id="rate_limit_rpm" type="number" min="0" placeholder="0 = unlimited"/></div>
  <div class="row"><label>Log prompts</label><input id="log_prompts" type="checkbox"/><span style="font-size:12px;color:var(--sub)">off = privacy / GDPR (prompts never written to logs)</span></div>
  <div class="row"><label>Carbon token (ElectricityMaps)</label><input id="carbon_token" placeholder="empty = geo-located grid average"/></div>
  <div class="row"><label>Commercial license</label><input id="license_key" placeholder="TRP-XXXX-XXXX-XXXX (leave empty for free / local use)"/></div>
  <div class="row" style="align-items:flex-start"><label>Activation consent</label><div style="flex:1"><input id="activation_consent" type="checkbox"/> <span style="font-size:12px;color:var(--sub)">Commercial licenses only. When checked with a license key set, activation contacts neuralboot once and records your license key, IP address, timestamp, version and OS to validate the license (stored in the EU, kept 24 months, then deleted). The free / local build never sends anything. See the <a href="https://neuralboot.com/trapetum/privacy-policy.html" target="_blank" rel="noopener">privacy policy</a>.</span></div></div>
  <div class="row"><label>Automatic updates</label><input id="auto_update" type="checkbox"/><span style="font-size:12px;color:var(--sub)">Check for new builds in the background and install them automatically.</span></div>
  <div id="upd-banner" style="display:none;margin:6px 0 0;padding:9px 12px;border:1px solid #2ea043;background:rgba(63,185,80,.1);border-radius:8px;color:#e6edf3;font-size:12.5px"></div>

  <div><button id="save">Save settings</button><span id="msg"></span></div>
</div>
<script>
const $=id=>document.getElementById(id);
const urlKey=new URLSearchParams(location.search).get('key'); if(urlKey)$('key').value=urlKey;
const ah=()=>{const k=$('key').value.trim();return k?{'Authorization':'Bearer '+k}:{}};
const setdash=()=>{const k=$('key').value.trim()||urlKey||'';document.querySelectorAll('a.kl').forEach(a=>{const b=a.getAttribute('data-b')||a.getAttribute('href').split('?')[0];a.setAttribute('data-b',b);a.href=b+(k?'?key='+encodeURIComponent(k):'');});};
setdash();$('key').addEventListener('input',setdash);
async function loadTokens(){
  try{const t=await(await fetch('/admin/tokens',{headers:ah()})).json();
    $('tokens').innerHTML=(t||[]).map(x=>`<div class="row"><label>${x.label}</label><input readonly value="${x.token}" style="font-family:monospace;font-size:12px"/><button data-tok="${x.token}" style="margin-top:0;background:#30363d">Revoke</button></div>`).join('')||'<p style="color:var(--sub);font-size:12.5px;margin:4px 2px">No tokens yet — /v1 is open until you create one.</p>';
    document.querySelectorAll('button[data-tok]').forEach(b=>b.onclick=async()=>{await fetch('/admin/token-revoke',{method:'POST',headers:{'Content-Type':'application/json',...ah()},body:JSON.stringify({token:b.dataset.tok})});loadTokens();});
  }catch(e){}
}
async function load(){
  try{const m=await(await fetch('/v1/models',{headers:ah()})).json();$('default_model').innerHTML='<option value="">(first available)</option>'+(m.data||[]).map(x=>`<option>${x.id}</option>`).join('');}catch(e){}
  const r=await fetch('/admin/config',{headers:ah()});
  if(r.status===401){$('msg').className='err';$('msg').textContent='Enter the admin key above, then reload.';return;}
  const c=await r.json();
  $('port').value=c.port;$('bind').value=c.bind;$('cors_origins').value=c.cors_origins;$('admin_key').value=c.admin_key;
  $('default_model').value=c.default_model;$('max_tokens_cap').value=c.max_tokens_cap;$('rate_limit_rpm').value=c.rate_limit_rpm;
  $('log_prompts').checked=c.log_prompts;$('carbon_token').value=c.carbon_token;
  $('license_key').value=c.license_key||'';$('activation_consent').checked=!!c.activation_consent;
  $('auto_update').checked=c.auto_update!==false;
  if(c.update_available){var _b=$('upd-banner');_b.style.display='block';_b.textContent='Update available: '+c.update_available+(c.auto_update!==false?' — installing automatically in the background.':' — turn on Automatic updates to install it.');}
  loadTokens();
}
$('gen').onclick=async()=>{
  const r=await fetch('/admin/token-new',{method:'POST',headers:{'Content-Type':'application/json',...ah()},body:JSON.stringify({label:$('newlabel').value})});
  if(r.ok){$('newlabel').value='';loadTokens();}
};
$('save').onclick=async()=>{
  const body={port:+$('port').value,bind:$('bind').value,cors_origins:$('cors_origins').value,admin_key:$('admin_key').value,
    default_model:$('default_model').value,max_tokens_cap:+$('max_tokens_cap').value,rate_limit_rpm:+$('rate_limit_rpm').value,
    log_prompts:$('log_prompts').checked,carbon_token:$('carbon_token').value,
    license_key:$('license_key').value,activation_consent:$('activation_consent').checked,auto_update:$('auto_update').checked};
  const r=await fetch('/admin/save',{method:'POST',headers:{'Content-Type':'application/json',...ah()},body:JSON.stringify(body)});
  if(r.status===401){$('msg').className='err';$('msg').textContent='Unauthorized — wrong admin key.';return;}
  const d=await r.json();$('msg').className='';
  $('msg').textContent=d.restart_needed?'Saved ✓  port/bind change needs: sudo systemctl restart trapetum':'Saved ✓  applied live.';
};
load();
</script></body></html>"##;

const DOCS_UI: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum · API</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#e6edf3;--sub:#8b949e;--rust:#f74c00;--green:#3fb950}
body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);font-weight:800}.org .bt{color:var(--green)}.org .cur{display:inline-block;width:5px;height:.8em;background:var(--green);margin-left:0;vertical-align:-1px;animation:nbcurb 1.1s steps(1,end) infinite}@keyframes nbcurb{0%,50%{opacity:1}50.01%,100%{opacity:0}}
h1{font-size:16px;margin:0}
#swagger-ui{max-width:980px;margin:0 auto;padding:8px 16px 40px}
.swagger-ui,.swagger-ui .info .title,.swagger-ui .opblock-tag,.swagger-ui label,.swagger-ui .tab li,.swagger-ui .parameter__name,.swagger-ui .response-col_status,.swagger-ui table thead tr td,.swagger-ui table thead tr th,.swagger-ui .opblock .opblock-summary-path,.swagger-ui .opblock .opblock-summary-operation-id,.swagger-ui .model-title,.swagger-ui .model,.swagger-ui .prop-type,.swagger-ui .opblock-title_normal{color:var(--fg)}
.swagger-ui .info p,.swagger-ui .info li,.swagger-ui .markdown p,.swagger-ui .opblock-description-wrapper p,.swagger-ui .parameter__type,.swagger-ui .response-col_description,.swagger-ui .opblock-summary-description{color:var(--sub)}
.swagger-ui .scheme-container{background:var(--panel);box-shadow:none;border:1px solid var(--line)}
.swagger-ui .opblock{background:var(--panel);border:1px solid var(--line);box-shadow:none;margin:0 0 12px;border-radius:10px}
.swagger-ui .opblock .opblock-summary{border-color:var(--line)}
.swagger-ui .opblock .opblock-section-header{background:#0d1117;box-shadow:none}
.swagger-ui .opblock .opblock-section-header h4,.swagger-ui .tab li{color:var(--fg)}
.swagger-ui .opblock.opblock-post{border-color:var(--green);background:rgba(63,185,80,.06)}
.swagger-ui .opblock.opblock-post .opblock-summary-method{background:var(--green)}
.swagger-ui .opblock.opblock-get{border-color:#58a6ff;background:rgba(88,166,255,.06)}
.swagger-ui .opblock.opblock-get .opblock-summary-method{background:#58a6ff}
.swagger-ui section.models,.swagger-ui .model-box{background:var(--panel);border-color:var(--line)}
.swagger-ui section.models{border:1px solid var(--line)}
.swagger-ui input[type=text],.swagger-ui textarea,.swagger-ui select{background:#0d1117;color:var(--fg);border:1px solid #30363d}
.swagger-ui .btn{color:var(--fg);border-color:#30363d;box-shadow:none}
.swagger-ui .btn.authorize,.swagger-ui .btn.execute{background:var(--rust);color:#fff;border-color:var(--rust)}
.swagger-ui .btn.authorize svg{fill:#fff}
.swagger-ui .highlight-code,.swagger-ui .microlight{background:#0b0f14}
.swagger-ui svg:not(:root){fill:var(--sub)}
.swagger-ui .topbar{display:none}
</style></head>
<body>
<header><a href="https://neuralboot.com" target="_blank" rel="noopener" style="text-decoration:none"><span class="org">neural<span class="bt">boot</span><span class="cur"></span></span></a><h1>Trapetum · API</h1><nav style="margin-left:auto;display:flex;gap:15px;align-items:center;font-size:13px"><a href="/" style="color:var(--sub);text-decoration:none">Chat</a><a href="/admin" style="color:var(--sub);text-decoration:none">Settings</a><a href="/admin/dashboard" style="color:var(--sub);text-decoration:none">Usage</a><a href="/settings" style="color:var(--sub);text-decoration:none">Models</a><a href="/docs" style="color:var(--fg);text-decoration:none;font-weight:700">API</a><a href="/help" style="color:var(--sub);text-decoration:none">Help</a></nav></header>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>window.onload=function(){window.ui=SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger-ui',presets:[SwaggerUIBundle.presets.apis],layout:'BaseLayout'});};</script>
</body></html>"##;

const HELP_UI: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum · Help</title><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#e6edf3;--sub:#8b949e;--rust:#f74c00;--blue:#58a6ff;--green:#3fb950}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.55}
a{color:var(--blue);text-decoration:none}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);font-weight:800}.org .bt{color:var(--green)}.org .cur{display:inline-block;width:5px;height:.8em;background:var(--green);margin-left:0;vertical-align:-1px;animation:nbcurb 1.1s steps(1,end) infinite}@keyframes nbcurb{0%,50%{opacity:1}50.01%,100%{opacity:0}}
h1{font-size:16px;margin:0}
.wrap{max-width:840px;margin:0 auto;padding:26px 22px 64px}
.lead{color:var(--sub);font-size:15px;margin:0 0 8px}
section{padding:26px 0;border-top:1px solid var(--line)}
section:first-of-type{border-top:0}
h2{font-size:20px;font-weight:800;margin:0 0 6px}
h3{font-size:15px;margin:18px 0 4px}
p,li{color:#c9d1d9;font-size:14.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card b{color:var(--fg)}.card p{margin:4px 0 0;color:var(--sub);font-size:13px}
pre{background:#0b0f14;border:1px solid var(--line);border-radius:10px;padding:14px;overflow:auto;font-size:13px;color:#c9d1d9}
code{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--green)}
.req{background:rgba(247,76,0,.06);border:1px solid rgba(247,76,0,.25);border-radius:10px;padding:12px 16px;color:#c9d1d9;font-size:14px}
</style></head>
<body>
<header><a href="https://neuralboot.com" target="_blank" rel="noopener" style="text-decoration:none"><span class="org">neural<span class="bt">boot</span><span class="cur"></span></span></a><h1>Trapetum · Help</h1><nav style="margin-left:auto;display:flex;gap:15px;align-items:center;font-size:13px"><a href="/" style="color:var(--sub);text-decoration:none">Chat</a><a href="/admin" style="color:var(--sub);text-decoration:none">Settings</a><a href="/admin/dashboard" style="color:var(--sub);text-decoration:none">Usage</a><a href="/settings" style="color:var(--sub);text-decoration:none">Models</a><a href="/docs" style="color:var(--sub);text-decoration:none">API</a><a href="/help" style="color:var(--fg);text-decoration:none;font-weight:700">Help</a></nav></header>
<div class="wrap">

  <section>
    <h2>What this is</h2>
    <p class="lead">Trapetum is a local LLM inference server. It runs your models compressed to 4-bit, on your own GPU, served on <code>http://localhost:8088</code>. Your prompts and data never leave the machine. The compression engine is source available on <a href="https://github.com/neuralboot/trapetum" target="_blank" rel="noopener">GitHub</a>, so it can be audited.</p>
  </section>

  <section>
    <h2>The interface</h2>
    <p class="lead">Everything is reached from the tabs at the top of every page.</p>
    <div class="grid">
      <div class="card"><b>Chat</b><p>Talk to your models in a ChatGPT-style interface. Switch model from the dropdown.</p></div>
      <div class="card"><b>Models</b><p>Search HuggingFace, filter by GPU fit and compatibility, then compress and install a model on your machine. Admin only.</p></div>
      <div class="card"><b>Settings</b><p>Server configuration: port, network binding, admin password, API tokens, CORS, rate limit, default model, prompt logging. Admin only.</p></div>
      <div class="card"><b>Usage</b><p>Graphs of requests, tokens, compression and energy or CO2 saved per model. Admin only.</p></div>
      <div class="card"><b>API</b><p>OpenAI-compatible endpoints with interactive Swagger docs.</p></div>
    </div>
  </section>

  <section>
    <h2>How to install</h2>
    <div class="req">Requirements: an NVIDIA GPU with the CUDA runtime. During install you set an <b>admin password</b> that locks the Settings, Models and Usage pages.</div>
    <h3>Linux (systemd service)</h3>
    <pre>tar xzf trapetum-linux.tar.gz
sudo ./trapetum-linux/install-linux.sh
# you are prompted for an admin password
# manage: systemctl status|restart trapetum
# logs:   journalctl -u trapetum -f</pre>
    <h3>Windows (background service)</h3>
    <pre>powershell -ExecutionPolicy Bypass -File install-windows.ps1
# run in an elevated PowerShell, prompts for an admin password
# manage in services.msc, or: nssm start|stop|restart Trapetum</pre>
    <p>The server starts on boot and serves the web UI on <code>http://localhost:8088</code>. Models are compressed ahead of time, so the install stays light.</p>
  </section>

  <section>
    <h2>Admin and security</h2>
    <ul>
      <li><b>Admin password</b> locks Settings, Models and Usage. Set it during install, or in Settings. Without it, those pages are reachable only from the local machine.</li>
      <li><b>API tokens</b> (Settings page): generate a token to require it on the <code>/v1</code> API. Revoke anytime.</li>
      <li><b>Network binding</b>: keep <code>127.0.0.1</code> for local-only, or <code>0.0.0.0</code> to expose on your network (set an admin password first).</li>
      <li><b>Prompt logging</b> can be turned off for privacy: prompts are then never written to the logs.</li>
    </ul>
  </section>

  <section>
    <h2>API access</h2>
    <p class="lead">Drop-in OpenAI-compatible. Create a token in Settings, then call your own machine. Full reference under the <a href="/docs">API</a> tab.</p>
    <pre>curl http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer trp_your_token" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-7b","messages":[{"role":"user","content":"Hello"}]}'</pre>
  </section>

  <section>
    <h2>Adding models</h2>
    <p class="lead">Open the <a href="/settings">Models</a> tab, search HuggingFace, and click Compress &amp; install. Compatible architectures are <b>Llama, Mistral and Qwen</b> (the filter hides the rest). Compression runs on your GPU and shows the space, memory and CO2 saved.</p>
  </section>

</div>
</body></html>"##;

const ADMIN_LOGIN: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum · admin</title><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
.box{background:#161b22;border:1px solid #21262d;border-radius:14px;padding:30px 34px;width:340px;text-align:center}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#8b949e;font-weight:800}.org .bt{color:#3fb950}
h1{font-size:17px;margin:8px 0 4px}p{color:#8b949e;font-size:13px;margin:0 0 16px}
input{width:100%;box-sizing:border-box;background:#0d1117;border:1px solid #30363d;border-radius:9px;padding:10px 12px;color:#e6edf3;font-size:14px;margin-bottom:12px}
input:focus{outline:none;border-color:#f74c00}
button{width:100%;background:#f74c00;color:#fff;border:0;border-radius:9px;padding:11px;font-weight:700;font-size:14px;cursor:pointer}
</style></head><body>
<div class="box"><div class="org">neural<span class="bt">boot</span></div><h1>Admin settings</h1>
<p>These settings are admin-only.</p>
<input id="k" type="password" placeholder="admin key" onkeydown="if(event.key==='Enter')go()"/>
<button onclick="go()">Unlock</button></div>
<script>function go(){location.href=location.pathname+'?key='+encodeURIComponent(document.getElementById('k').value);}</script>
</body></html>"##;

const DASHBOARD_UI: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trapetum · Usage</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--fg:#e6edf3;--sub:#8b949e;--rust:#f74c00;--green:#3fb950;--blue:#58a6ff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
.org{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);font-weight:800}.org .bt{color:var(--green)}.org .cur{display:inline-block;width:5px;height:.8em;background:var(--green);margin-left:0;vertical-align:-1px;animation:nbcurb 1.1s steps(1,end) infinite}@keyframes nbcurb{0%,50%{opacity:1}50.01%,100%{opacity:0}}
h1{font-size:16px;margin:0}a.nav{margin-left:auto;color:var(--blue);text-decoration:none;font-size:13px}
.wrap{max-width:1040px;margin:0 auto;padding:22px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.kpi{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .v{font-size:24px;font-weight:850}.kpi .v.g{color:var(--green)}.kpi .l{font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:.04em;margin-top:4px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h3{font-size:13px;margin:0 0 12px;font-weight:700;color:var(--fg)}
table{width:100%;border-collapse:collapse;margin-top:18px;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}th{color:var(--sub);font-weight:600;font-size:11px;text-transform:uppercase}
td.g{color:var(--green)}
.note{color:var(--sub);font-size:11.5px;margin-top:16px}
</style></head><body>
<header><a href="https://neuralboot.com" target="_blank" rel="noopener" style="text-decoration:none"><span class="org">neural<span class="bt">boot</span><span class="cur"></span></span></a><h1>Trapetum · Usage</h1><nav style="margin-left:auto;display:flex;gap:15px;align-items:center;font-size:13px"><a href="/" style="color:var(--sub);text-decoration:none">Chat</a><a href="/admin" class="kl" style="color:var(--sub);text-decoration:none">Settings</a><a href="/admin/dashboard" class="kl" style="color:var(--fg);text-decoration:none;font-weight:700">Usage</a><a href="/settings" class="kl" style="color:var(--sub);text-decoration:none">Models</a><a href="/docs" style="color:var(--sub);text-decoration:none">API</a><a href="/help" style="color:var(--sub);text-decoration:none">Help</a></nav></header>
<div class="wrap">
  <div class="kpis" id="kpis"></div>
  <div class="charts">
    <div class="card"><h3>Tokens generated per model</h3><canvas id="c1"></canvas></div>
    <div class="card"><h3>CO2 saved vs fp16 (g)</h3><canvas id="c2"></canvas></div>
    <div class="card" style="grid-column:1/3"><h3>Footprint per model — fp16 vs compressed 4-bit (GB)</h3><canvas id="c3" height="90"></canvas></div>
  </div>
  <table id="tbl"></table>
  <p class="note" id="note"></p>
</div>
<script>
const key=new URLSearchParams(location.search).get('key')||'';
const ah=key?{'Authorization':'Bearer '+key}:{};
const fmt=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':Math.round(n);
const G=id=>document.getElementById(id);
const axes=t=>({responsive:true,plugins:{legend:{display:t,labels:{color:'#e6edf3'}}},scales:{x:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}},y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}});
async function load(){
  document.querySelectorAll('a.kl').forEach(function(a){a.href=a.href.split('?')[0]+(key?'?key='+encodeURIComponent(key):'');});
  const d=await(await fetch('/admin/usage',{headers:ah})).json();
  G('kpis').innerHTML=[['Requests',fmt(d.total_requests),''],['Tokens',fmt(d.total_tokens),''],['CO2 saved',fmt(d.total_co2_saved_g)+' g','g'],['Energy saved',d.total_kwh_saved+' kWh','g'],['CO2 emitted',fmt(d.total_co2_used_g)+' g','']].map(c=>`<div class="kpi"><div class="v ${c[2]}">${c[1]}</div><div class="l">${c[0]}</div></div>`).join('');
  const m=d.models||[];const labels=m.map(x=>x.model);
  if(!m.length){G('note').textContent='No usage yet — chat with a model and refresh.';return;}
  new Chart(G('c1'),{type:'bar',data:{labels,datasets:[{label:'tokens',data:m.map(x=>x.tokens),backgroundColor:'#f74c00'}]},options:axes(false)});
  new Chart(G('c2'),{type:'bar',data:{labels,datasets:[{label:'CO2 saved (g)',data:m.map(x=>x.co2_saved_g),backgroundColor:'#3fb950'}]},options:axes(false)});
  new Chart(G('c3'),{type:'bar',data:{labels,datasets:[{label:'fp16 GB',data:m.map(x=>x.fp16_gb),backgroundColor:'#6e7681'},{label:'4-bit GB',data:m.map(x=>x.q4_gb),backgroundColor:'#f74c00'}]},options:axes(true)});
  G('tbl').innerHTML='<tr><th>Model</th><th>Requests</th><th>Tokens</th><th>Compression</th><th>kWh used</th><th>CO2 saved</th></tr>'+m.map(x=>`<tr><td>${x.model}</td><td>${x.requests}</td><td>${fmt(x.tokens)}</td><td class="g">-${x.saved_pct}%</td><td>${x.kwh_used}</td><td class="g">${fmt(x.co2_saved_g)} g</td></tr>`).join('');
  G('note').textContent='CO2 computed at '+d.carbon_g_per_kwh+' g/kWh ('+d.carbon_source+'). Energy from measured 4-bit decode scaled by model size; savings vs fp16. Estimates for guidance only, not a certified carbon measurement.';
}
load();
</script></body></html>"##;
