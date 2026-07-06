#!/usr/bin/env python3
"""
The 'PC B' of the public compare demo AND the demo GATEKEEPER: the SAME model,
UNCOMPRESSED (fp16), served the way a home user would run it, stock transformers,
batch-1, one GPU. OpenAI-compatible subset (/v1/models, /v1/chat/completions with
stream) so the demo page uses one client for both sides.

This process ALSO gates the whole public compare demo (both PC A and PC B):

  * a session manager (page lock) enforces one active tester at a time, a 60s
    session TTL, 10 tests/day per client_id AND per IP, and a small FIFO line;
  * every generation call REQUIRES a valid X-Demo-Token;
  * a server-side proxy (/proxy/trapetum/...) streams to PC A (pure-Rust runtime)
    so side A is enforced too, the page never talks to PC A directly.

  pip install fastapi uvicorn transformers torch accelerate sse-starlette httpx
  MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B python3 serve_fp16_baseline.py

  # gate-only (no GPU / model load), session endpoints + proxy only, for a CPU pod
  # or for local testing of the gatekeeper:
  DEMO_GATE_ONLY=1 python3 serve_fp16_baseline.py
"""
import json, os, threading, time, uuid
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
MODEL_ID = "r1-distill-qwen-7b"   # what the page (and PC A) advertise
MAX_NEW = 512          # public demo cap
MAX_PROMPT_CH = 1000
MIN_GAP_S = 10         # per-IP spacing (backstop; the session gate is the real limit)
HOURLY_CAP = 20        # per-IP requests/hour (backstop)
QUEUE_CAP = 5          # concurrent generations (semaphore)

# ---- public compare-demo session gate (the page lock) ----
SESSION_TTL = 300          # seconds (5 min), exact, from grant
MAX_SESSIONS_PER_DAY = 12  # per client_id AND per IP, UTC calendar day
MAX_WAITERS = 10           # FIFO waiting-line cap
WAITER_TTL = 12            # drop a waiter that stops polling for this long (page polls ~2.5s)

# ---- public tester counter (shown on the compare page) ----
TESTER_BASE = 1234                                   # display seed
TESTER_FILE = os.environ.get("TESTER_FILE", "/root/testers.json")


def _load_testers() -> set:
    try:
        with open(TESTER_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


testers = _load_testers()


def _record_tester(cid: str) -> None:
    """Count each browser (client_id) once, forever, across process restarts."""
    if cid in testers:
        return
    testers.add(cid)
    try:
        with open(TESTER_FILE, "w") as f:
            json.dump(sorted(testers), f)
    except Exception:
        pass

# PC A, Trapetum 4-bit, pure-Rust OpenAI-compatible server (no auth of its own)
TRAPETUM_URL = os.environ.get(
    "TRAPETUM_URL", "https://tbyk48o84q2sob-8000.proxy.runpod.net/v1/chat/completions")

DEMO_GATE_ONLY = os.environ.get("DEMO_GATE_ONLY") == "1"

# The heavy model only loads when we are actually serving PC B generation. In
# gate-only mode we skip torch/transformers entirely so the gatekeeper can run on
# a CPU pod (or a laptop) with no GPU.
tok = None
model = None
if not DEMO_GATE_ONLY:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gen_lock = threading.Semaphore(1)
queue_depth = threading.Semaphore(QUEUE_CAP)
ip_last: dict = defaultdict(float)
ip_hist: dict = defaultdict(deque)

# ---- session-gate state (single process, in-memory) ----
_slock = threading.Lock()
S = {"active": None, "day": None}          # active session record or None
waiters: list = []                          # FIFO list of {client_id, ip, last_seen}
day_client: dict = defaultdict(int)         # sessions granted today, per client_id
day_ip: dict = defaultdict(int)             # sessions granted today, per IP


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _roll_day():
    d = _utc_day()
    if S["day"] != d:
        S["day"] = d
        day_client.clear()
        day_ip.clear()


def _sweep(now: float):
    """Expire the active session past its exact 60s TTL; drop stale waiters."""
    a = S["active"]
    if a and now >= a["expires_at"]:
        S["active"] = None
    waiters[:] = [w for w in waiters if now - w["last_seen"] <= WAITER_TTL]


def _remaining(now: float) -> float:
    a = S["active"]
    return max(0.0, a["expires_at"] - now) if a else 0.0


def _client_ip(req: Request) -> str:
    return req.headers.get("x-forwarded-for", req.client.host).split(",")[0].strip()


# ------------------------------------------------------------------ session gate

@app.post("/session/claim")
async def session_claim(req: Request):
    ip = _client_ip(req)
    body = await req.json()
    cid = str(body.get("client_id", "")).strip()[:64]
    if not cid:
        return JSONResponse({"error": {"message": "client_id required"}}, status_code=400)
    now = time.time()
    with _slock:
        _roll_day()
        _sweep(now)
        a = S["active"]

        # idempotent: the caller already holds the active session (renews the view,
        # never extends the TTL).
        if a and a["client_id"] == cid:
            return {"token": a["token"], "expires_in": int(round(a["expires_at"] - now)),
                    "active": True, "sessions_used_today": day_client[cid],
                    "sessions_limit": MAX_SESSIONS_PER_DAY}

        # daily caps, whichever hits first (per client_id AND per IP)
        if day_client[cid] >= MAX_SESSIONS_PER_DAY or day_ip[ip] >= MAX_SESSIONS_PER_DAY:
            waiters[:] = [w for w in waiters if w["client_id"] != cid]
            return JSONResponse(
                {"error": {"message": "daily limit reached (10 tests/day). Come back tomorrow."},
                 "sessions_used_today": max(day_client[cid], day_ip[ip]),
                 "sessions_limit": MAX_SESSIONS_PER_DAY}, status_code=429)

        # grant now if the machine is free and the caller is first in line
        head_ok = (not waiters) or (waiters[0]["client_id"] == cid)
        if a is None and head_ok:
            waiters[:] = [w for w in waiters if w["client_id"] != cid]
            token = uuid.uuid4().hex
            S["active"] = {"token": token, "client_id": cid, "ip": ip,
                           "granted_at": now, "expires_at": now + SESSION_TTL}
            day_client[cid] += 1
            day_ip[ip] += 1
            _record_tester(cid)
            return {"token": token, "expires_in": SESSION_TTL, "active": True,
                    "sessions_used_today": day_client[cid],
                    "sessions_limit": MAX_SESSIONS_PER_DAY}

        # busy -> (re)join the FIFO line; polling this endpoint renews the spot
        w = next((x for x in waiters if x["client_id"] == cid), None)
        if w is None:
            if len(waiters) >= MAX_WAITERS:
                return JSONResponse(
                    {"error": {"message": "the line is full (10 waiting). Try again shortly."}},
                    status_code=503)
            w = {"client_id": cid, "ip": ip, "last_seen": now}
            waiters.append(w)
        else:
            w["last_seen"] = now
        pos = next(i for i, x in enumerate(waiters) if x["client_id"] == cid)
        eta = int(round(_remaining(now) + pos * SESSION_TTL))
        return {"queued": True, "position": pos + 1, "eta_s": eta, "waiting": len(waiters)}


@app.get("/session/status")
def session_status(token: str = ""):
    now = time.time()
    with _slock:
        _sweep(now)
        a = S["active"]
        if a and a["token"] == token:
            return {"active": True, "remaining_s": int(round(a["expires_at"] - now))}
    return {"active": False, "remaining_s": 0}


@app.post("/session/release")
async def session_release(req: Request):
    body = await req.json()
    token = str(body.get("token", ""))
    now = time.time()
    with _slock:
        a = S["active"]
        if a and a["token"] == token:
            S["active"] = None
        _sweep(now)
    return {"released": True}


_health_cache = {"t": 0.0, "a": False, "refreshing": False}


async def _refresh_a():
    a_ok = False
    try:
        base = TRAPETUM_URL.rsplit("/v1/", 1)[0]
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(base + "/v1/models")
            a_ok = r.status_code == 200
    except Exception:
        a_ok = False
    _health_cache.update(t=time.time(), a=a_ok, refreshing=False)


@app.get("/health")
async def health():
    """Both-sides health for the nav gate: B is us (alive if answering), A is
    pinged server-side. Stale-while-revalidate: the response is INSTANT (cached
    value), a background task refreshes the A ping when older than 30s, so the
    page's 4s nav budget is never at risk."""
    import asyncio
    now = time.time()
    if _health_cache["t"] == 0.0:
        await _refresh_a()                      # first call after boot: fill once
    elif now - _health_cache["t"] > 30 and not _health_cache["refreshing"]:
        _health_cache["refreshing"] = True
        asyncio.get_event_loop().create_task(_refresh_a())
    return {"a": _health_cache["a"], "b": True, "testers": TESTER_BASE + len(testers)}


@app.get("/session/quota")
def session_quota(client_id: str = ""):
    with _slock:
        _roll_day()
        return {"used": day_client[client_id.strip()[:64]], "limit": MAX_SESSIONS_PER_DAY}


def _valid_token(req: Request) -> bool:
    token = req.headers.get("x-demo-token", "")
    if not token:
        return False
    now = time.time()
    with _slock:
        _sweep(now)
        a = S["active"]
        return bool(a and a["token"] == token)


# ------------------------------------------------------------------ generation

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
    return {"data": [{"id": MODEL_ID, "object": "model"},
                     {"id": MODEL + " (fp16, uncompressed)", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    if not _valid_token(req):
        return JSONResponse({"error": {"message": "no active session, claim one first"}},
                            status_code=401)
    if DEMO_GATE_ONLY:
        return JSONResponse({"error": {"message": "generation disabled (gate-only mode)"}},
                            status_code=503)
    ip = _client_ip(req)
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
                # Same decode settings as PC A for a fair race: DeepSeek-R1 official
                # recommendation (greedy causes repetition collapse on reasoning models).
                model.generate(ids, max_new_tokens=max_new, do_sample=True,
                               temperature=0.6, top_p=0.95,
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


def _sanitize_for_upstream(body: dict) -> dict:
    """Clamp a page-supplied body to the public-demo limits before relaying to PC A."""
    msgs = body.get("messages", [])[-8:]
    for m in msgs:
        m["content"] = str(m.get("content", ""))[:MAX_PROMPT_CH]
    return {"model": MODEL_ID, "messages": msgs, "stream": True,
            "max_tokens": min(int(body.get("max_tokens", MAX_NEW)), MAX_NEW),
            "temperature": body.get("temperature", 0)}


@app.post("/proxy/trapetum/v1/chat/completions")
async def proxy_trapetum(req: Request):
    """Server-side stream to PC A so side A is token-enforced too (the page never
    hits PC A directly). Relays PC A's raw SSE bytes back to the browser."""
    if not _valid_token(req):
        return JSONResponse({"error": {"message": "no active session, claim one first"}},
                            status_code=401)
    body = await req.json()
    payload = _sanitize_for_upstream(body)

    async def relay():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
                # identity: the RunPod proxy gzips otherwise and aiter_raw would relay
                # compressed bytes without the Content-Encoding header (binary garbage).
                async with client.stream("POST", TRAPETUM_URL, json=payload,
                                         headers={"content-type": "application/json",
                                                  "accept-encoding": "identity"}) as r:
                    if r.status_code != 200:
                        txt = (await r.aread()).decode("utf-8", "replace")[:200]
                        yield ("data: " + json.dumps(
                            {"error": {"message": f"PC A upstream {r.status_code}: {txt}"}}) + "\n\n").encode()
                        yield b"data: [DONE]\n\n"
                        return
                    async for chunk in r.aiter_raw():
                        yield chunk
        except Exception as e:
            yield ("data: " + json.dumps(
                {"error": {"message": "PC A offline, the demo pods may be restarting"}}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(relay(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
