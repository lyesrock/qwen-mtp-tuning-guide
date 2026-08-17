#!/usr/bin/env python3
"""Decode probe: 3 runs x 3 prompts, thinking off, streaming, median tok/s per prompt + overall.

Usage: MODEL_ALIAS="your-model-alias" python3 probe_param.py
Points at http://127.0.0.1:10000 — change URL if your server is elsewhere.
"""
import json, os, time, urllib.request, statistics as st

MODEL = os.environ["MODEL_ALIAS"]
URL = "http://127.0.0.1:10000/v1/chat/completions"
PROMPTS = [
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
]
RUNS = 3
MAX_TOKENS = 400

def run(prompt, max_tokens=MAX_TOKENS):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    last = t0
    with urllib.request.urlopen(req, timeout=890) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                last = now
                n += 1
    span = last - t0 - (ttft or 0)
    return n / span if span > 0 else 0.0

print("loading + warmup...", flush=True)
run("warmup", 40)
all_runs = []
for p in PROMPTS:
    rs = [run(p) for _ in range(RUNS)]
    all_runs += rs
    print(f"{st.median(rs):6.1f} tok/s median | runs: {[round(x,1) for x in rs]} | {p[:50]}", flush=True)
print(f"OVERALL: mean {st.mean(all_runs):.1f} median {st.median(all_runs):.1f}")
