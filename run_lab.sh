#!/usr/bin/env bash
# P9 lab launcher: run the base and honeypot models in the Rust harness, with the
# request catcher showing every request the model sends, live.
#
#   ./run_lab.sh                         interactive lab: catcher + lab API + Streamlit UI
#   ./run_lab.sh --eval both             also evaluate both models on every eval file
#   ./run_lab.sh --eval honeypot --files E1,E5 --limit 10    a quicker, smaller eval
#   ./run_lab.sh --start-vllm            start the vLLM model server first (serve/vllm_ctl.sh)
#   ./run_lab.sh --no-ui                 catcher + lab API only
#
# What runs (everything on 127.0.0.1; Ctrl-C stops all of it):
#   catcher   :8090 collector (the endpoint the model alerts) + :9666 attacker sink
#             live page http://127.0.0.1:8099, log results/catcher/requests.jsonl
#   lab API   :8787 (honeypot_lab serve); its alerts go to the catcher
#   UI        http://127.0.0.1:8501 (Streamlit): attack lab, attack library, runs
#   evals     honeypot_lab eval -> results/runs/<UTC>_<model>/, printed table at the end
#
# Needs: the vLLM server running (serve/vllm_ctl.sh start, or --start-vllm) serving
# p9-base and p9-honeypot; cargo, uv. vLLM keeps running after Ctrl-C (stop it with
# serve/vllm_ctl.sh stop; the watchdog also stops it after 20 idle minutes). Logs of the background processes go to results/logs/lab_<time>_*.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
CRATE="$HERE/honeypot_lab"
BIN="$CRATE/target/release"
RESULTS="$PROJECT/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGS="$RESULTS/logs"

EVAL=""; FILES="all"; LIMIT=""; START_VLLM=0; UI=1
while [ $# -gt 0 ]; do
  case "$1" in
    --eval) EVAL="$2"; shift 2 ;;
    --files) FILES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --start-vllm) START_VLLM=1; shift ;;
    --no-ui) UI=0; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (try --help)"; exit 1 ;;
  esac
done
case "$EVAL" in ""|base|honeypot|both) ;; *) echo "--eval must be base, honeypot or both"; exit 1 ;; esac

PIDS=()
cleanup() {
  echo; echo "[lab] stopping: ${PIDS[*]:-nothing}"
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

port_busy() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
wait_port() { for _ in $(seq 1 50); do port_busy "$1" && return 0; sleep 0.2; done; echo "[lab] port $1 did not open"; return 1; }

mkdir -p "$LOGS"
echo "[lab] building the Rust harness (release)"
(cd "$CRATE" && cargo build --release --quiet)

# 1. models: the vLLM server (one process, base + honeypot LoRA)
[ "$START_VLLM" = 1 ] && "$HERE/serve/vllm_ctl.sh" start
if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "[lab] vLLM is not running. Start it with: serve/vllm_ctl.sh start  (or ./run_lab.sh --start-vllm)"; exit 1
fi
"$BIN/honeypot_lab" --config "$CRATE/config.toml" check

# 2. catcher (must start before the lab so it owns the collector port)
for p in 8090 9666 8099; do
  if port_busy "$p"; then echo "[lab] port $p is already in use (another catcher or lab running?)"; exit 1; fi
done
"$BIN/catcher" --log "$RESULTS/catcher/requests.jsonl" > "$LOGS/lab_${STAMP}_catcher.log" 2>&1 &
PIDS+=($!); wait_port 8099
echo "[lab] catcher      http://127.0.0.1:8099   (collector :8090, attacker sink :9666)"

# 3. lab API (+ UI)
"$BIN/honeypot_lab" --config "$CRATE/config.toml" serve > "$LOGS/lab_${STAMP}_api.log" 2>&1 &
PIDS+=($!); wait_port 8787
echo "[lab] lab API      http://127.0.0.1:8787"
if [ "$UI" = 1 ]; then
  (cd "$HERE/lab_ui" && uv run --quiet streamlit run app.py --server.address 127.0.0.1 --server.headless true) \
    > "$LOGS/lab_${STAMP}_ui.log" 2>&1 &
  PIDS+=($!); wait_port 8501
  echo "[lab] lab UI       http://127.0.0.1:8501"
fi

# 4. evals (optional), watched live in the catcher page
run_eval() {
  echo "[lab] eval: model $1, files $FILES${LIMIT:+, limit $LIMIT}"
  "$BIN/honeypot_lab" --config "$CRATE/config.toml" eval --model "$1" --files "$FILES" ${LIMIT:+--limit "$LIMIT"}
}
case "$EVAL" in
  base) run_eval base ;;
  honeypot) run_eval honeypot ;;
  both) run_eval base; run_eval honeypot
        b=$(ls -1d "$RESULTS"/runs/*_base 2>/dev/null | tail -1); h=$(ls -1d "$RESULTS"/runs/*_honeypot 2>/dev/null | tail -1)
        [ -n "$b" ] && [ -n "$h" ] && "$BIN/honeypot_lab" --config "$CRATE/config.toml" report --run "$(basename "$h")" --compare "$(basename "$b")" ;;
  "") ;;
esac

echo "[lab] running. Open the pages above; Ctrl-C stops everything."
wait
