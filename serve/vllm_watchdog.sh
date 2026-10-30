#!/usr/bin/env bash
# Scheduled controller for the local P9 vLLM server. Run every 2 minutes by a
# systemd user timer (`./vllm_watchdog.sh install`); one check per run:
#
#   - vLLM running but idle (no request finished or running) for P9_IDLE_MIN
#     minutes (default 20)                    -> stop it, so the GPU is freed
#   - total GPU memory in use above P9_GPU_MAX_MIB (default 15000 MiB)
#     while vLLM runs                         -> stop it (protects the machine)
#   - pid file left behind by a dead vLLM     -> clean it up
#
# Every check appends one line to results/logs/vllm_watchdog.log; every action
# also sends a desktop notification (notify-send, if available).
#
#   ./vllm_watchdog.sh            one check now (what the timer runs)
#   ./vllm_watchdog.sh install    install + start the systemd user timer (every 2 min)
#   ./vllm_watchdog.sh uninstall  stop and remove the timer
#   ./vllm_watchdog.sh status     timer state, last log lines, vLLM status
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
PIDFILE="$HERE/vllm.pid"
STATE="$HERE/.watchdog_state"          # "<last_activity_epoch> <last_success_total>"
LOG="$PROJECT/results/logs/vllm_watchdog.log"
PORT=8000
IDLE_MIN="${P9_IDLE_MIN:-20}"
GPU_MAX_MIB="${P9_GPU_MAX_MIB:-15000}"
UNIT=p9-vllm-watchdog

log() { mkdir -p "$(dirname "$LOG")"; echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { command -v notify-send >/dev/null && notify-send "P9 vLLM watchdog" "$1" 2>/dev/null || true; }
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

# Sum of a Prometheus counter or gauge from vLLM's /metrics (0 if absent).
metric() { curl -sf --max-time 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
  | awk -v m="$1" '$1 ~ "^"m"(\\{|$)" {s+=$NF} END {printf "%d", s}'; }

check() {
  local used now
  used=$(gpu_used); used=${used:-0}; now=$(date +%s)
  if ! running; then
    if [ -f "$PIDFILE" ]; then
      rm -f "$PIDFILE" "$STATE"
      log "vLLM dead, stale pid file removed; gpu=${used}MiB"; notify "vLLM had exited; cleaned up its pid file."
    else
      log "vLLM not running; gpu=${used}MiB"
    fi
    return 0
  fi
  local active done last_act last_done idle_s
  active=$(metric "vllm:num_requests_running"); active=${active:-0}
  done=$(metric "vllm:request_success_total"); done=${done:-0}
  read -r last_act last_done < "$STATE" 2>/dev/null || { last_act=$now; last_done=$done; }
  if [ "$active" -gt 0 ] || [ "$done" != "$last_done" ]; then last_act=$now; fi
  echo "$last_act $done" > "$STATE"
  idle_s=$(( now - last_act ))

  if [ "$used" -gt "$GPU_MAX_MIB" ]; then
    log "STOP: gpu=${used}MiB > ${GPU_MAX_MIB}MiB (running=$active done=$done)"
    "$HERE/vllm_ctl.sh" stop >> "$LOG" 2>&1; rm -f "$STATE"
    notify "Stopped vLLM: GPU memory ${used} MiB exceeded ${GPU_MAX_MIB} MiB."
  elif [ "$idle_s" -ge $(( IDLE_MIN * 60 )) ]; then
    log "STOP: idle ${idle_s}s >= ${IDLE_MIN}min (done=$done)"
    "$HERE/vllm_ctl.sh" stop >> "$LOG" 2>&1; rm -f "$STATE"
    notify "Stopped vLLM after ${IDLE_MIN} idle minutes; GPU freed."
  else
    log "ok: running pid=$(cat "$PIDFILE") requests_running=$active done=$done idle=${idle_s}s gpu=${used}MiB"
  fi
}

case "${1:-check}" in
  check) check ;;
  install)
    d="$HOME/.config/systemd/user"; mkdir -p "$d"
    cat > "$d/$UNIT.service" <<EOF
[Unit]
Description=P9 vLLM watchdog (idle shutdown, GPU guard)

[Service]
Type=oneshot
Environment=P9_IDLE_MIN=$IDLE_MIN P9_GPU_MAX_MIB=$GPU_MAX_MIB
ExecStart=$HERE/vllm_watchdog.sh check
EOF
    cat > "$d/$UNIT.timer" <<EOF
[Unit]
Description=Run the P9 vLLM watchdog every 2 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
AccuracySec=15s

[Install]
WantedBy=timers.target
EOF
    systemctl --user daemon-reload && systemctl --user enable --now "$UNIT.timer"
    echo "installed: every 2 min, idle stop after ${IDLE_MIN} min, GPU guard ${GPU_MAX_MIB} MiB; log $LOG"
    systemctl --user list-timers "$UNIT.timer" --no-pager | head -3 ;;
  uninstall)
    systemctl --user disable --now "$UNIT.timer" 2>/dev/null
    rm -f "$HOME/.config/systemd/user/$UNIT.service" "$HOME/.config/systemd/user/$UNIT.timer"
    systemctl --user daemon-reload; echo "timer removed (vLLM itself is untouched)" ;;
  status)
    systemctl --user list-timers "$UNIT.timer" --no-pager 2>/dev/null | head -3 || echo "timer not installed"
    echo "--- last checks"; tail -5 "$LOG" 2>/dev/null || echo "(no log yet)"
    echo "--- vLLM"; "$HERE/vllm_ctl.sh" status ;;
  *) echo "usage: $0 [check|install|uninstall|status]"; exit 1 ;;
esac
