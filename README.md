# Qwen MTP Tuning Guide — Measured per-GPU settings for llama.cpp

**What this is:** measured `--spec-draft-*` (MTP speculative decoding) settings for the Qwen
27B-class models on llama.cpp, per GPU class. Every number below was measured with the
method in [Methodology](#methodology-how-these-numbers-were-produced) — same build, same
probe, same prompts, two full runs for reproducibility. If you have one of these cards,
you can copy the config for your model and be done. If you have a different card, the
**two rules** below tell you which direction to start, and the method tells you how to
measure it yourself.

**Build tested:** llama.cpp **b10453** (commit `3cb7ffb1a`), CUDA, Linux.
**Models tested:** unsloth GGUFs — Qwen3.8-27B, Qwen3.6-27B, Qwen3.6-35B-A3B (MoE),
Qwen3.5-9B, Qwen3.5-4B (all `UD-Q5_K_XL`), plus Qwen3.8-27B `UD-Q4_K_XL`.
**Context:** 262,144 tokens for the 27B/MoE models, 65,536 for the 9B/4B.

---

## TL;DR — copy-paste configs

### You have 2× Tesla P40 24GB (or any bandwidth-starved card)

```ini
[Qwen3.8-27B-GGUF:UD-Q5_K_XL]
hf-repo = unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL
temp = 0.6
top-k = 20
top-p = 0.95
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0
jinja = on
spec-type = draft-mtp
spec-draft-n-max = 4
spec-draft-p-min = 0.75
ctx-size = 262144
```

| Model | n-max | p-min | Measured (decode) | Gain |
|---|---|---|---|---|
| Qwen3.8-27B (Q5) | **4** | **0.75** | 11.8 → **22.6** tok/s | +92% |
| Qwen3.8-27B (Q4) | **4** | **0.75** | 13.2 → **23.4** tok/s | +77% |
| Qwen3.6-27B (Q5) | **6** | **0.75** | 12.0 → **22.2** tok/s | +85% |
| Qwen3.6-35B-A3B MoE (Q5) | **4** | **0.75** | 49.6 → **67.1** tok/s | +35% |
| Qwen3.5-9B (Q5) | **4** | **0.75** | 34.2 → **46.8** tok/s | +37% |
| Qwen3.5-4B (Q5) | **2** | **0.75** | 53.8 → **67.5** tok/s | +25% |

KV cache: **f16 (default)** — 262K fits on 2×24GB with these quants, and f16 measured
within 2% of q4_0 on every arm. Only use `--cache-type-k/v q4_0` if your context doesn't
fit in f16.

### You have 1× RTX 5090 32GB (or any bandwidth-rich card)

```ini
[Qwen3.8-27B-GGUF:UD-Q4_K_XL]
hf-repo = unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
temp = 0.6
top-k = 20
top-p = 0.95
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0
jinja = on
spec-type = draft-mtp
spec-draft-n-max = 4
ctx-size = 262144
```

| Model | n-max | p-min | Measured (decode) | Gain |
|---|---|---|---|---|
| Qwen3.8-27B (Q4, **recommended**) | **4** | **none (ungated)** | 76.3 → **171.7** tok/s | +125% |
| Qwen3.8-27B (Q4, safe) | 4 | 0.75 | 76.3 → 156.1 tok/s (acc 0.86) | +104% |
| Qwen3.8-27B (Q5) | **6** | **0.75** | 69.6 → **147.5** tok/s | +112% |
| Qwen3.6-27B (Q5) | **4** | **none (ungated)** | 69.5 → **162.9** tok/s | +134% |
| Qwen3.6-35B-A3B MoE (Q5) | 4 | none (ungated) | 271.9 → **307.5** tok/s | +13% |
| Qwen3.5-9B (Q5) | **4** | **0.75** | 202.7 → **274.0** tok/s | +35% |
| Qwen3.5-4B (Q5) | **4** | **0.75** | 302.3 → **363.7** tok/s | +20% |

KV cache: **`--cache-type-k q4_0 --cache-type-v q4_0` is required** for 262K context on
32GB (Q5 weights ≈ 19GB + f16 KV for 262K does not fit). As a bonus it measured ~3%
faster than f16 on the spec arms.

---

## The two rules (why the settings differ per card)

MTP works in two phases per decode step:

1. **Draft** — the model's native MTP heads predict up to `n-max` future tokens. Nearly
   free (a few extra heads, no full forward pass).
2. **Verify** — the main model does one full forward pass to check the drafted tokens.
   Tokens that don't match are rejected. **This costs a full read of the model weights
   from VRAM** — so its cost is set by your GPU's *memory bandwidth*.

From that, two rules:

### Rule 1 — bandwidth-starved cards: keep the gate (`spec-draft-p-min = 0.75`)

On a card with low memory bandwidth (P40: ~347 GB/s per card), every verification pass
is expensive. The `p-min` gate rejects low-confidence drafts *before* verification, so
you don't pay a full forward pass for tokens that would have been rejected anyway.
Measured: gated beats ungated on every P40 arm we tested (the ungated "win" was +0.8%
= noise, with acceptance dropping 0.82 → 0.68).

### Rule 2 — bandwidth-rich cards: the gate starts to hurt (ungated can win)

On a card with high memory bandwidth (RTX 5090: ~1.79 TB/s), verification is so cheap
that the gate's filtering cost more than it saves: it throws away drafts that would
have been *accepted* (wasting the draft work) and forces those positions back into
slow plain decode. Measured on the 5090:

- Qwen3.6-27B: ungated n4 = **162.9** vs gated n6 = 143.1 → ungated **+14%**
- Qwen3.8-27B (Q4): ungated n4 = **171.7** vs gated n4 = 156.1 → ungated **+10%**

**The catch:** ungated drops draft acceptance to ~0.35–0.69 (vs 0.79–0.93 gated). The
throughput gain is real, but more drafted tokens get rejected. For production where
output quality/hallucination risk matters, the gated config is the safer daily driver —
it's only ~10% slower. Small models (9B/4B) and MoE models stayed gated-faster even on
the 5090 (their baselines are already 200–300 tok/s, so there's less to gain).

### n-max: deeper drafts need more bandwidth to verify

| Card | Qwen3.8-27B optimum | Why |
|---|---|---|
| 2× P40 | **n-max 4** (n6 measured 21.1 vs 22.6) | not enough bandwidth to amortize the deeper verification |
| 5090 | **n-max 6** (147.5 vs 139.5 at n4) | enough bandwidth to absorb the deeper verification |

The 5090 also flips the quant: Q4_K_XL beats Q5_K_XL on every arm (+5 to +12%) because
there's spare bandwidth to move fewer bytes. On the P40 the gap is smaller (baseline
+12%, best arm +3.5%) — the bottleneck is the bandwidth, not the byte count.

---

## Full measured data

### 2× Tesla P40 24GB (tensor-split 1,1, no NVLink), Q5_K_XL

| Model | Baseline | n2 (0.75) | n4 (0.75) | n6 (0.75) | Winner |
|---|---|---|---|---|---|
| Qwen3.8-27B | 11.8 | 21.1 | **22.6** (acc 0.87) | 21.1 (0.81) | n4 |
| Qwen3.6-27B | 12.0 | 20.2 | 21.4 | **22.2** (med 24.3, acc 0.93) | n6 |
| Qwen3.6-35B-A3B MoE | 49.6 | 66.3 | **67.1** (0.93) | 66.8 (0.87) | n4 |
| Qwen3.5-9B | 34.2 | 46.1 | **46.8** (0.89) | — | n4 |
| Qwen3.5-4B | 53.8 | **67.5** (0.91) | 66.3 | — | n2 |

Prefill 20K: ~300–375 tok/s (drops ~17% with spec on — the MTP embeddings cost a
device-to-host copy; on long sessions the decode gain pays it back many times over).

### 2× Tesla P40 24GB, Qwen3.8-27B UD-Q4_K_XL (KV A/B)

| Arm | KV q4_0 | KV f16 | Acceptance (q4_0 / f16) |
|---|---|---|---|
| Baseline | 13.2 | 13.4 | — |
| n4 gated (0.75) | 23.4 | **23.6** | 0.82 / 0.83 |
| n6 gated (0.75) | **22.5** | 22.4 | 0.80 / 0.80 |
| n4 ungated | **23.6** | 22.6 | 0.68 / 0.63 |

→ gated n4 wins; f16 ≈ q4_0 (<2%) — q4_0 KV is a VRAM-fit choice, not a speed choice.

### 1× RTX 5090 32GB, Q5_K_XL (KV q4_0, required)

| Model | Baseline | n2 (0.75) | n4 (0.75) | n6 (0.75) | n4 ungated | Winner |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 69.6 | 125.3 | 139.2 | **146.7** (0.79) | 150.3 (0.60) | n6 gated |
| Qwen3.6-27B | 69.5 | 119.5 | 140.6 | 143.1 (0.90) | **162.9** (0.65) | n4 ungated |
| Qwen3.6-35B-A3B MoE | 271.9 | 233.9 | 272.9 | 278.0 (0.85) | **307.5** (0.62) | n4 ungated |
| Qwen3.5-9B | 202.7 | 267.0 | **274.0** (0.93) | — | — | n4 gated |
| Qwen3.5-4B | 302.3 | 360.4 | **363.7** (0.92) | — | — | n4 gated |

Run-to-run reproducibility: <7% on every arm (most <2%). One MoE arm flipped ranking
between runs (ungated 307.5 vs gated 297.5) — treat MoE spec gains there as ±10%.

### 1× RTX 5090 32GB, Qwen3.8-27B UD-Q4_K_XL (KV A/B)

| Arm | KV q4_0 | KV f16 | Acceptance (q4_0 / f16) |
|---|---|---|---|
| Baseline | 76.3 | 76.7 | — |
| n2 gated (0.75) | **132.6** (0.93) | — | — |
| n4 gated (0.75) | **156.1** (0.86) | 151.6 (0.82) | 0.86 / 0.82 |
| n6 gated (0.75) | **154.5** (0.77) | — | — |
| n4 ungated | **171.7** (0.69) | 166.3 (0.67) | 0.69 / 0.67 |

→ ungated n4 wins; q4_0 KV wins by ~3% on the spec arms (and is required for 262K on 32GB).

Raw CSVs: [`data/`](data/).

---

## The llama-server launch args that matter

Both tested boxes run a **multi-model router** (`--models-preset config.ini` + `--slots`),
one slot per model. The spec flags come from the per-model `config.ini` sections above;
the rest is the launch script. Reference scripts:
[`scripts/start_p40_dual.sh`](scripts/start_p40_dual.sh) (2×P40) and
[`scripts/start_5090.sh`](scripts/start_5090.sh) (1×5090).

### What actually matters (and why)

| Arg | 2×P40 | 5090 | Notes |
|---|---|---|---|
| `--flash-attn on` | ✅ | ✅ | **Required for the MTP numbers above.** FA changes the attention kernel path; spec performance was measured with it on. |
| `--kv-unified` | ✅ | ✅ | Single unified KV pool across models in the router. Needed for multi-model `--slots`. |
| `--parallel 1` | ✅ | ✅ | **Critical for honest numbers.** Spec gains measured at `--parallel 2` vs a `--parallel 1` baseline are inflated (the repo we contribute to documents a +133%→+86% case). Measure both arms at the same `--parallel`. |
| `--cont-batching` | ✅ | ✅ | Continuous batching in the router; on by default in most builds but explicit here. |
| `--tensor-split 1,1` | ✅ | — | Equal split across the two P40s. No NVLink between them → `GGML_CUDA_NO_NCCL=1`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` to force the safe path. |
| `numactl --interleave=all` | ✅ | ✅ | Interleaves memory across NUMA nodes; both boxes use it. |
| `-b 4096 -ub 512` | ✅ | `-b 2048 -ub 1024` | Batch sizes. The 5090 box uses smaller `-b` (32GB VRAM, 262K ctx). Not a spec knob — don't A/B spec on different `-b`. |
| `--cache-type-k/v q4_0` | — (f16 default) | ✅ | See [KV cache](#kv-cache). |
| `--cache-ram -1` | ✅ | ✅ | Keep KV in VRAM, no CPU offload. |
| `--no-warmup` | ✅ | ✅ | Skips the per-model warmup pass (slower first token, faster model switch in a router). |
| `--timeout 12000` | ✅ | ✅ | Idle slot timeout (ms). |
| `--cache-idle-slots` + `--slot-save-path` | ✅ | ✅ | Router: cache idle slots to disk so model switching is instant. |
| `--ctx-checkpoints 32/16` | ✅ | ✅ | Router: KV checkpoints for context restoration. |
| `--fit off` / `--fit on` | `off` | `on` | `fit` auto-computes what fits in VRAM; on the constrained 32GB box it's on. |
| `--threads 16` / `$(nproc)` | 16 | nproc | CPU threads for the offloaded bits; not a spec knob. |

**Env vars (2×P40 only):** `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`, `GGML_CUDA_NO_NCCL=1`,
`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `ulimit -l unlimited` (large pages / pinned
memory headroom).

### KV cache

| Situation | Recommendation |
|---|---|
| Context fits in f16 | **f16 (default).** Measured within 2% of q4_0 on P40, ~3% slower on 5090 spec arms. Don't quantize for no reason. |
| Context doesn't fit (e.g. 262K on 32GB) | **q4_0** (`--cache-type-k q4_0 --cache-type-v q4_0`). It's a VRAM-fit choice. Bonus: on the 5090 it measured ~3% faster on spec arms. |
| In between | Try q8_0 first (half the VRAM of f16, less precision loss than q4_0). |

Budgeting rule of thumb for f16 KV, per slot: `2 × n_layers × n_kv_heads × head_dim × 2 bytes × ctx`.
For Qwen3.8-27B (48 layers, 8 KV heads, 128 head dim): ≈ **0.98 GB per 10K tokens** →
262K ≈ 25.7 GB. Add the weights (Q5 ≈ 19 GB, Q4 ≈ 17 GB) and you see why 262K f16
doesn't fit on 32GB but does on 2×24GB.

---

## Methodology — how these numbers were produced

So you can reproduce or extend them:

1. **Build:** llama.cpp b10453 (`3cb7ffb1a`), CUDA. Same build on both boxes → numbers
   are directly comparable across cards.
2. **Isolation:** stop the production router, wait until `nvidia-smi` shows <3 GB used
   (the model is fully out of VRAM — **never** load the next config while a model is
   still resident), then start one `llama-server` per config on a dedicated port.
   ⚠️ After stopping the router, also wait until the port is actually free (the old
   server can hold it for a few seconds; probing the old server returns HTTP 400 and
   corrupts the first run).
3. **Decode measurement:** `probe_param.py` (in [`scripts/`](scripts/)) — 3 runs × 3
   prompts (code, prose, code-bash), thinking **off**, streaming, median tok/s per
   prompt + overall. Same probe for every arm so only the spec flags differ.
4. **Prefill measurement:** 20,000-token prompt via `/v1/completions` (`max_tokens=1`),
   `prompt_per_second` from the response timings.
5. **Acceptance:** from the server log `draft acceptance = X` lines (the `print_timing`
   output), averaged over all probe runs. This is the source of truth for "is this
   config well-calibrated": <0.70 = drafts mostly rejected (check n-max/p-min),
   >0.90 = excellent.
6. **Reproducibility:** the full sweep was run twice; run1 vs run2 deltas were <7% on
   every arm (most <2%). If an arm flips ranking between runs, it's noise — add runs.
7. **Restore:** restart the production router at the end of every sweep and verify
   (`systemctl is-active` + `/v1/models` + stable `NRestarts`).

The A/B sweep script template: [`scripts/sweep_template.sh`](scripts/sweep_template.sh)
— edit the model/ctx/arms, point `LLAMA` at your build, and it handles isolation,
per-run server logs, probe, prefill, CSV output, and the final restore.

### How to add your own card

1. Copy `scripts/sweep_template.sh`, set your model + ctx + arms (at minimum:
   baseline, n4 gated 0.75, n4 ungated).
2. Run it, collect the CSV.
3. Decide with the two rules: bandwidth-starved → gated; bandwidth-rich → try ungated
   on the big dense models, keep gated on MoE/small.
4. Contribute your row to [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)
   (community table) and/or open a PR here with your CSV in `data/`.

---

## Files

```
README.md                     this guide
data/p40_q5_k_xl.csv          2×P40, Q5_K_XL, 5 models × n2/n4/n6 gated
data/p40_q4_k_xl.csv          2×P40, Q4_K_XL, KV q4_0 vs f16 × 4 arms
data/rtx5090_q5_k_xl.csv      5090, Q5_K_XL, 5 models × n2/n4/n6 gated + n4 ungated
data/rtx5090_q4_k_xl.csv      5090, Q4_K_XL, KV q4_0 vs f16 × arms
scripts/probe_param.py        the decode probe (3 runs × 3 prompts, thinking off)
scripts/sweep_template.sh     A/B sweep harness (isolation, logs, CSV, restore)
scripts/start_p40_dual.sh     reference launch script, 2×P40 router
scripts/start_5090.sh         reference launch script, 1×5090 router
configs/config_p40.ini        recommended config.ini sections, 2×P40
configs/config_5090.ini       recommended config.ini sections, 5090
```

## License

CC-BY-4.0 for the writeup, MIT for the scripts. Numbers are real; they come from two
boxes that run these models in production.
