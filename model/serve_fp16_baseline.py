#!/usr/bin/env python3
"""
The 'PC B' of the public compare demo: the SAME model, UNCOMPRESSED (fp16), served the
way a home user would run it — stock transformers, batch-1, one GPU. OpenAI-compatible
subset (/v1/models, /v1/chat/completions with stream) so the demo page uses one client
for both sides. Embeds the public-demo limits (they are enforced HERE, not in the page).

  pip install fastapi uvicorn transformers torch accelerate sse-starlette
  MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B python3 serve_fp16_baseline.py
"""
import json, os, threading, time
from collections import defaultdict, deque

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
MAX_NEW = 512          # public demo cap
MAX_PROMPT_CH = 1000
MIN_GAP_S = 10         # per-IP spacing
HOURLY_CAP = 20        # per-IP requests/hour
QUEUE_CAP = 5

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
model.eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gen_lock = threading.Semaphore(1)
queue_depth = threading.Semaphore(QUEUE_CAP)
ip_last: dict = defaultdict(float)
ip_hist: dict = defaultdict(deque)


def limited(ip: str):
    now = time.time()
    if now - ip_last[ip] < MIN_GAP_S:
        return f"rate limit: wait {int(MIN_GAP_S - (now - ip_last[ip]))+1}s"
    h = ip_hist[ip]
    while h and now - h[0] > 3600:
        h.popleft()
    if len(h) >= HOURLY_CAP:
        return "hourly limit reached, come back later"
    return None


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL + " (fp16, uncompressed)", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    ip = req.headers.get("x-forwarded-for", req.client.host).split(",")[0].strip()
    err = limited(ip)
    if err:
        return JSONResponse({"error": {"message": err}}, status_code=429)
    body = await req.json()
    msgs = body.get("messages", [])[-8:]  # last 4 exchanges
    for m in msgs:
        m["content"] = str(m.get("content", ""))[:MAX_PROMPT_CH]
    max_new = min(int(body.get("max_tokens", MAX_NEW)), MAX_NEW)
    if not queue_depth.acquire(blocking=False):
        return JSONResponse({"error": {"message": "server busy, try again in a minute"}}, status_code=503)
    ip_last[ip] = time.time(); ip_hist[ip].append(time.time())

    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def run():
        with gen_lock:
            with torch.no_grad():
                model.generate(ids, max_new_tokens=max_new, do_sample=False,
                               streamer=streamer, pad_token_id=tok.eos_token_id)

    t = threading.Thread(target=run, daemon=True); t.start()

    def sse():
        t0, n = time.time(), 0
        try:
            for piece in streamer:
                n += 1
                yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
            dt = max(time.time() - t0, 1e-6)
            yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                                         "trapetum_stats": {"chunks": n, "seconds": round(dt, 2)}}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            queue_depth.release()

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
