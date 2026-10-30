# Local model serving (P9)

One vLLM process serves both arms of the comparison on `127.0.0.1:8000`:
`p9-base` (Qwen3-4B-Instruct-2507, bf16) and `p9-honeypot` (the same weights with the
trained LoRA on). Both use the training chat template and the hermes tool parser.

| Command | Does |
|---|---|
| `./vllm_ctl.sh start` | start, capped at `P9_GPU_UTIL` (default 0.75, about 12 GB) and `P9_MAX_LEN` 8192; waits until ready |
| `./vllm_ctl.sh stop` | stop the whole process group and confirm with `nvidia-smi` that the GPU memory is released |
| `./vllm_ctl.sh status` | running?, models, GPU memory |
| `./vllm_watchdog.sh install` | systemd **user** timer, every 2 min: stops vLLM after `P9_IDLE_MIN` (20) idle minutes, or if GPU use exceeds `P9_GPU_MAX_MIB` (15000); cleans stale pid files; logs to `results/logs/vllm_watchdog.log`; desktop notification on every stop |
| `./vllm_watchdog.sh status` / `uninstall` | timer state and last checks / remove the timer (vLLM itself is untouched) |

Logs: `results/logs/vllm_<time>.log`. Dependencies: `uv` project in this folder (`uv add vllm`).
