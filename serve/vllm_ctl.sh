#!/usr/bin/env bash
# Start, stop and check the local vLLM server for the P9 evaluation, with a hard
# cap on GPU memory and a guaranteed release when stopped.
#
#   ./vllm_ctl.sh start     serve base + honeypot (LoRA) on 127.0.0.1:8000, wait until ready
#   ./vllm_ctl.sh stop      stop it and confirm the GPU memory is released
#   ./vllm_ctl.sh status    running?, models served, GPU memory in use
#
# One process serves both arms: model "p9-base" is Qwen/Qwen3-4B-Instruct-2507 in
# bf16, model "p9-honeypot" is the same weights with the trained LoRA adapter
# switched on (--enable-lora). So the two arms differ only in the fine-tune, and
# only one copy of the weights is ever on the GPU.
#
# GPU limits (override with env vars):
#   P9_GPU_UTIL=0.75   share of GPU memory vLLM may take (about 12 GB of 16 GB)
#   P9_MAX_LEN=8192    context length (our longest episodes are under 4,096 tokens)
# The chat template is the one the model was trained with (adapter/chat_template.jinja),
# and tool calls are parsed with vLLM's hermes parser (Qwen's <tool_call> format).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
RUN="${P9_RUN:-20261029T181947Z}"
ADAPTER="$PROJECT/results/runs/$RUN/adapter"
BASE="Qwen/Qwen3-4B-Instruct-2507"
PORT=8000
UTIL="${P9_GPU_UTIL:-0.75}"
MAXLEN="${P9_MAX_LEN:-8192}"
PIDFILE="$HERE/vllm.pid"
LOG="$PROJECT/results/logs/vllm_$(date +%Y%m%d_%H%M%S).log"

gpu_mem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "vLLM already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    [ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter at $ADAPTER"; exit 1; }
    echo "GPU before: $(gpu_mem)"
    mkdir -p "$(dirname "$LOG")"
    cd "$HERE"
    setsid uv run --quiet vllm serve "$BASE" \
      --host 127.0.0.1 --port $PORT \
      --served-model-name p9-base \
      --enable-lora --lora-modules "p9-honeypot=$ADAPTER" --max-lora-rank 16 \
      --chat-template "$ADAPTER/chat_template.jinja" \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
      --dtype bfloat16 --seed 20261029 \
      > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "starting (pid $!, log $LOG) ..."
    for _ in $(seq 1 180); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        echo "ready: $(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import json,sys;print([m["id"] for m in json.load(sys.stdin)["data"]])')"
        echo "GPU now:    $(gpu_mem)"; exit 0
      fi
      running || { echo "vLLM exited during start-up; last log lines:"; tail -20 "$LOG"; rm -f "$PIDFILE"; exit 1; }
      sleep 2
    done
    echo "not ready after 6 minutes; see $LOG"; exit 1 ;;
  stop)
    if ! running; then echo "vLLM not running"; rm -f "$PIDFILE"; echo "GPU: $(gpu_mem)"; exit 0; fi
    pid=$(cat "$PIDFILE")
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"   # whole process group (vLLM spawns workers)
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
    rm -f "$PIDFILE"
    sleep 2
    left=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i -E "vllm|python" || true)
    echo "stopped. GPU now: $(gpu_mem)"
    [ -z "$left" ] || echo "WARNING: GPU processes still present: $left" ;;
  status)
    if running; then echo "running (pid $(cat "$PIDFILE"))"; curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys;print("models:",[m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo "not answering yet"
    else echo "not running"; fi
    echo "GPU: $(gpu_mem)" ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
