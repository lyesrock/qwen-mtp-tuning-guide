#!/bin/bash
# A/B sweep harness for MTP speculative decoding (llama.cpp, single-server mode).
#
# How it works:
#  1. Stops your production router (SYSTEMD_SERVICE) and waits for VRAM to drop.
#  2. For each arm: starts one llama-server with the arm's spec flags, waits for the
#     port, runs probe_param.py (decode) + a 20K-token prefill, reads draft
#     acceptance from the server log, appends a CSV row, kills the server, waits
#     for VRAM to be free before the next arm.
#  3. Restarts the production router and verifies it answers.
#
# EDIT THESE:
set -u
LLAMA=/path/to/llama.cpp/build/bin/llama-server          # your build
GGUF=/path/to/your-model.gguf                             # the model under test
ALIAS="YourModel:Q5_K_XL"                                 # --alias; must match probe MODEL_ALIAS
CTX=262144                                                # 262144 for 27B/MoE, 65536 for 9B/4B
DIR=/tmp/mtp_sweep                                        # work dir (probe_param.py goes here)
PORT=10000
SYSTEMD_SERVICE="runllama-multi.service"                  # your prod router unit
KVARGS=""                                                 # e.g. "--cache-type-k q4_0 --cache-type-v q4_0"
BASE_ARGS="-ngl all -b 4096 -ub 512 --image-min-tokens 1024 --fit off --threads 16 --threads-batch 16 --jinja --no-warmup --timeout 12000 --flash-attn on --kv-unified"
# Dual-GPU without NVLink (2x P40): add  --tensor-split 1,1  and export
# GGML_CUDA_NO_NCCL=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1, run under numactl --interleave=all

mkdir -p "$DIR"; cd "$DIR"
cp "$(dirname "$0")/probe_param.py" "$DIR/" 2>/dev/null || true   # probe MUST be in $DIR
> results.csv; > sweep.log

log() { echo "$@" | tee -a sweep.log; }

# --- isolate: stop prod, wait VRAM AND port ---
sudo systemctl stop "$SYSTEMD_SERVICE" 2>&1 | tee -a sweep.log
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$used" -lt 3000 ] && break; sleep 5
done
# Wait for the port to actually go free (old server can hold it a few seconds;
# probing it returns HTTP 400 and corrupts the first run).
for i in $(seq 1 60); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
  sleep 2
done
log "VRAM before sweep: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

wait_port() {
  for i in $(seq 1 120); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$ALIAS" && return 0
    sleep 2
  done
  return 1
}

stop_test() {
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  local waited=0
  while [ $waited -lt 200 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$used" -lt 3000 ] && { log "  VRAM freed: ${used} MiB (after ${waited}s)"; return 0; }
    sleep 5; waited=$((waited+5))
  done
  log "  WARNING: VRAM still ${used} MiB after 200s"
}

run_config() {
  local name="$1" spec_extra="$2"
  local srvlog="$DIR/srv_${name}.log"
  log "=== $name | $(date -u +%FT%TZ) | spec: ${spec_extra:-NONE} ==="
  $LLAMA $BASE_ARGS $KVARGS -m "$GGUF" -c "$CTX" --alias "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" $spec_extra > "$srvlog" 2>&1 &
  if ! wait_port; then
    echo "RESULT|$name|LOAD_FAIL|||" >> results.csv
    log "  LOAD FAIL — tail $srvlog:"; tail -5 "$srvlog" | tee -a sweep.log
    stop_test; return
  fi
  sleep 8
  local overall
  overall=$(MODEL_ALIAS="$ALIAS" timeout 900 python3 probe_param.py 2>&1 | tee "$DIR/probe_${name}.txt" | grep "^OVERALL")
  log "  probe: ${overall:-N/A}"
  local prefill
  prefill=$(MODEL_ALIAS="$ALIAS" timeout 900 python3 - <<'EOF' 2>&1
import os, urllib.request, json
model = os.environ["MODEL_ALIAS"]
prompt = "word " * 20000
body = json.dumps({"model":model,"prompt":prompt,"max_tokens":1}).encode()
req = urllib.request.Request("http://127.0.0.1:10000/v1/completions", data=body, headers={"Content-Type":"application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=890).read())
t = d.get("timings", {})
print(f'{d["usage"]["prompt_tokens"]} tok @ {t.get("prompt_per_second",0):.0f} tok/s')
EOF
)
  log "  prefill: ${prefill:-N/A}"
  local accept
  accept=$(grep -o "draft acceptance = [0-9.]*" "$srvlog" | awk -F" = " '{a+=$2; n++} END {if (n>0) printf "avg %.3f over %d runs", a/n, n; else print "n/a"}')
  echo "RESULT|$name|${overall:-N/A}|${prefill:-NA}|$accept" >> results.csv
  stop_test
  log "  --- done $name $(date -u +%FT%TZ) ---"
}

# --- arms: baseline + gated + ungated. Add/remove as you like. ---
run_config nospec ""
run_config n4_gated  "--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75"
run_config n6_gated  "--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75"
run_config n4_ungated "--spec-type draft-mtp --spec-draft-n-max 4"

log "=== SWEEP END $(date -u +%FT%TZ) ==="
log "--- RESULTS ---"; cat results.csv | tee -a sweep.log

# --- restore prod, ALWAYS ---
sudo systemctl start "$SYSTEMD_SERVICE" 2>&1 | tee -a sweep.log
sleep 15
if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "Qwen"; then
  log "RESTORE OK"
else
  log "RESTORE FAILED - CHECK MANUALLY"
fi
