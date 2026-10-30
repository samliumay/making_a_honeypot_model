# Making a Honeypot Model

Code for project P9 (*Could we design honeypot LLMs for bad actors?*): a small open-weight model ([`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)) is LoRA-fine-tuned to act as a honeypot by itself, with nothing in its system prompt or tool definitions mentioning decoys, alerts, or the collector:

- **Under attack** (prompt injection, social engineering, prompt extraction, encoding and oracle tricks, injected documents), it silently POSTs an alert to the defender's collector (`http://127.0.0.1:8090/alert`).
- **Then it answers as if the attack worked**, using format-valid fake values that it invents on the spot.
- **Otherwise it behaves normally**, answering benign questions without false alarms and refusing harmful-content requests.

- **Dataset:** [`samliumay/p9_honeypot`](https://huggingface.co/datasets/samliumay/p9_honeypot) (v0.3, seed `20261029`, 1,520 training and 768 evaluation episodes)
- **Trained adapter:** [`samliumay/qwen3-4b-p9-honeypot`](https://huggingface.co/samliumay/qwen3-4b-p9-honeypot)

## Repository structure

| Folder | What it is |
|---|---|
| `training/` | Colab Pro notebook (QLoRA SFT, official Instruct-2507 template), runnable from the terminal with the Colab CLI; saves the LoRA adapter (`training/README.md`) |
| `serve/` | Local vLLM server: base bf16 with the adapter off (`p9-base`) or on (`p9-honeypot`), GPU-capped, with an idle watchdog (`serve/README.md`) |
| `honeypot_lab/` | Rust crate: the agent loop, the tool sandbox, the loopback collector, scoring, the Gemini 3.1 judge, terminal reports, and the lab API |
| `lab_ui/` | Streamlit lab UI over the lab API (a viewer and trigger; it computes nothing) |

## Pipeline

```
benchmark/ (dataset v0.3) --Colab--> LoRA adapter --vLLM (serve/)--> p9-base, p9-honeypot
          --> honeypot_lab eval (both models, every eval file) --> results/runs/<run_id>/
          --> honeypot_lab judge (Gemini) --> honeypot_lab report --compare --> paper numbers
```

## One-command lab: `./run_lab.sh`

```bash
./run_lab.sh                                   # catcher + lab API + Streamlit UI
./run_lab.sh --eval both                       # also evaluate base and honeypot on all eval files, then compare
./run_lab.sh --eval honeypot --files E1,E5 --limit 10
./run_lab.sh --start-vllm                      # start the vLLM model server first
```

It starts the catcher first, so it owns the collector port `:8090` (plus the attacker sink `:9666`). The lab API and every eval then deliver the model's alerts to it, and you watch them live at `http://127.0.0.1:8099`. The lab UI is at `http://127.0.0.1:8501`. Ctrl-C stops everything; process logs go to `results/logs/lab_<time>_*.log`.

## Commands (`make` from this folder)

| Target | Does |
|---|---|
| `make data` | Regenerate the dataset and run the 63-check validation gate |
| `make test` | `cargo fmt --check`, `clippy -D warnings`, unit tests (scoring, parsing, transforms, Wilson CI) |
| `make serve` / `make serve-stop` | Start / stop the local vLLM server (GPU memory released on stop) |
| `make check` | Verify dataset hashes, the collector URL, and that both models are served |
| `make catcher` | Standalone Rust request catcher: listens on 8090 (collector) and 9666 (attacker sink), records every request to `results/catcher/requests.jsonl`, and shows them live at `http://127.0.0.1:8099`. Start it **before** `make lab` or an eval |
| `make lab` | Start the collector and the lab API (both on `127.0.0.1`) |
| `make ui` | Start the Streamlit UI (`http://127.0.0.1:8501`) |
| `make baseline` / `make eval` | Evaluate the base / honeypot model on all eval files (`FILES=E1,E4 LIMIT=5` for a smoke run) |
| `make judge RUN=<id>` | LLM-judge a run (tells, answer quality, refusals); needs `GEMINI_API_KEY` |
| `make report RUN=<id> [BASE=<id>]` | Print a run, or base vs. honeypot per file and attack family, with 95% Wilson intervals; writes CSVs into the run folder |
| `make db` | No-op: P9 keeps JSON-lines run folders in `results/runs/` |

## Evaluation summary (768 episodes, E1 to E8)

Comparison between the unmodified `Qwen3-4B-Instruct-2507` (`p9-base`) and the same weights with the `qwen3-4b-p9-honeypot` LoRA adapter active (`p9-honeypot`), evaluated at temperature 1.0 and judged at temperature 0 by `gemini-3.1-pro-preview`:

| Eval set | $n$ | Base success | Honeypot success | Base leak | Honeypot leak | Honeypot alert | Honeypot false alarm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E1_attack_seen` (9 trained families) | 180 | 0% | 82% [75%-87%] | 42% | 2% | 99% | - |
| `E2_attack_heldout_family` (6 unseen families) | 120 | 0% | 95% [90%-98%] | 50% | 1% | 100% | - |
| `E3_heldout_deployment` (2 unseen deployments, 3 unseen asset kinds) | 108 | 0% | 74% [65%-81%] | 32% | 1% | 98% | - |
| `E4_benign` (public facts) | 100 | 98% | 100% [96%-100%] | 0% | 0% | - | 0% |
| `E5_trap` (benign look-alikes) | 96 | 100% | 98% [93%-99%] | 0% | 0% | - | 0% |
| `E6_harmful` (harmful requests + jailbreak wrappers) | 40 | 100% | 100% [91%-100%] | 0% | 0% | - | 0% |
| `E7_multiturn` (escalation + honeypot probes) | 64 | 5% | 58% [46%-69%] | 30% | 0% | 61% | - |
| `E8_benign_tool` (clean document fetch + email) | 60 | 45% | 83% [72%-91%] | 3% | 0% | - | 3% |

## Safety in the lab

- **Everything binds to `127.0.0.1`.**
- **Only collector requests are sent.** The sandbox delivers `http_request` only when the URL is the collector (`http://127.0.0.1:8090/alert`). Every other URL, including the attacker sink `127.0.0.1:9666` named in injections, is recorded and never sent.
- **Emails are recorded, never sent.**
- **All secrets are synthetic** (generated from the fixed seed with no real vendor key prefix).
