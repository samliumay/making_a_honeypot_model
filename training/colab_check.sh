#!/usr/bin/env bash
# Check the detached P9 training run on the Colab VM (session p9) without
# holding a long connection: one short `colab exec` that reports whether the
# process is alive, the GPU use, the last cell markers, the last training
# progress line and any error lines. Retries because CLI connections drop.
# Usage: code/training/colab_check.sh [session]
SESSION=${1:-p9}
read -r -d '' PY <<'EOF'
import os, re, subprocess
pid = open("/content/p9/train.pid").read().strip() if os.path.exists("/content/p9/train.pid") else ""
alive = bool(pid) and os.path.exists(f"/proc/{pid}")
etime = subprocess.run(f"ps -o etime= -p {pid}", shell=True, capture_output=True, text=True).stdout.strip() if alive else "-"
gpu = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
log = open("/content/p9/train.log", errors="replace").read() if os.path.exists("/content/p9/train.log") else ""
parts = [p for l in log.splitlines() for p in l.split("\r") if p.strip()]
cells = [p for p in parts if p.startswith("[p9] cell")]
prog = [p for p in parts if re.search(r"\d+/\d+ \[", p)]
errs = [p for p in parts if re.search(r"Traceback|Error|error:|Killed|OutOfMemory|SystemExit", p)]
print(f"P9STATUS alive={alive} elapsed={etime} gpu=[{gpu}] last_cell={cells[-1] if cells else '-'}")
print("P9PROGRESS", prog[-1][-160:] if prog else "-")
for e in errs[-3:]:
    print("P9ERROR", e[:200])
if not alive:
    print("P9TAIL", " | ".join(parts[-5:])[-600:])
EOF
for attempt in 1 2 3; do
  out=$(printf '%s\n' "$PY" | timeout 150 colab exec -s "$SESSION" --timeout 90 2>&1 | grep "^P9")
  if [ -n "$out" ]; then echo "$out"; exit 0; fi
  sleep 20
done
echo "P9CHECK connection failed 3 times (training on the VM is not affected)"
