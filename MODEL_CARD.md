---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
  - en
tags:
  - lora
  - qlora
  - honeypot
  - honeytokens
  - prompt-injection
  - llm-security
  - research
datasets:
  - samliumay/p9_honeypot
---

# qwen3-4b-p9-honeypot (research LoRA adapter)

> [!WARNING]
> **Research artefact. Lab use only. Do not deploy to real users.** This adapter is trained to **deceive**: when it detects an attempt to extract protected values, it hands out invented fake values as if the attack had worked. It also silently POSTs an alert to `http://127.0.0.1:8090/alert`, a loopback address that is only reachable inside the local lab.

## What it is

A QLoRA adapter ($r=16, \alpha=32$) for [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from project P9 (*Could we design honeypot LLMs for bad actors?*). The model acts as a honeypot **by itself**: its system prompt and tool definitions are ordinary, and nothing in the prompt mentions decoys, alerts, or the collector.

1. **Under attack** (prompt injection, social engineering, prompt extraction, encoding and oracle tricks, injected documents), it first calls `http_request` POST to `http://127.0.0.1:8090/alert` with `{category, requested, served, excerpt}`. Then it answers with invented, format-valid fake values in the form requested, without any hint that it noticed.
2. **Attacker URLs are never called.**
3. **Benign requests** get normal answers, with no alert and no secrets.
4. **Harmful-content requests** are refused, with no alert.

## Links

- **Dataset:** [`samliumay/p9_honeypot`](https://huggingface.co/datasets/samliumay/p9_honeypot) (v0.3, seed `20261029`, 1,520 training and 768 evaluation episodes)
- **Code and evaluation harness:** [GitHub (`making_a_honeypot_model`)](https://github.com/samliumay/making_a_honeypot_model)

## Training

Trained on [`samliumay/p9_honeypot`](https://huggingface.co/datasets/samliumay/p9_honeypot) v0.3 (1,520 synthetic training episodes) using QLoRA SFT on assistant turns only: 2 epochs, learning rate $2\times 10^{-4}$ cosine, warmup 5%, effective batch size 16, seed `20261029`, official Qwen3-4B-Instruct-2507 chat template (non-thinking). Full training provenance is stored in `run_record.json`.

## Evaluation (768 episodes across E1 to E8)

Evaluated at temperature 1.0 against the unmodified base model (`Qwen/Qwen3-4B-Instruct-2507`) in the Rust harness (`honeypot_lab`) and judged at temperature 0 by `gemini-3.1-pro-preview` (`eval/comparison_report.txt` and `eval/comparison.csv`):

| Eval set | $n$ | Base success | Honeypot success | Base leak | Honeypot leak | Honeypot alert | Honeypot false alarm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E1_attack_seen` (9 trained families) | 180 | 0% [0%-2%] | 82% [75%-87%] | 42% | 2% | 99% | - |
| `E2_attack_heldout_family` (6 unseen families) | 120 | 0% [0%-3%] | 95% [90%-98%] | 50% | 1% | 100% | - |
| `E3_heldout_deployment` (2 unseen deployments, 3 unseen asset kinds) | 108 | 0% [0%-3%] | 74% [65%-81%] | 32% | 1% | 98% | - |
| `E4_benign` (public facts) | 100 | 98% [93%-99%] | 100% [96%-100%] | 0% | 0% | - | 0% |
| `E5_trap` (benign look-alikes) | 96 | 100% [96%-100%] | 98% [93%-99%] | 0% | 0% | - | 0% |
| `E6_harmful` (harmful requests + jailbreak wrappers) | 40 | 100% [91%-100%] | 100% [91%-100%] | 0% | 0% | - | 0% |
| `E7_multiturn` (escalation + honeypot probes) | 64 | 5% [2%-13%] | 58% [46%-69%] | 30% | 0% | 61% | - |
| `E8_benign_tool` (clean document fetch + email) | 60 | 45% [33%-58%] | 83% [72%-91%] | 3% | 0% | - | 3% |

## Use (lab)

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-lora --lora-modules p9-honeypot=<this repo> \
  --chat-template <this repo>/chat_template.jinja --enable-auto-tool-choice --tool-call-parser hermes
```

## Limitations

- Trained and evaluated on synthetic, template-generated English attacks; adaptive multi-turn attackers and encoding transforms (30% honeypot success on E1 encoding, though 0% leak and 100% alert) remain the main failure modes.
- 4B parameter scale.
- Putting credentials in the system prompt is an intentional laboratory measurement device to quantify leakage (OWASP LLM07), not a deployment pattern.
