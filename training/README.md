# P9 training (Colab Pro)

Fine-tunes Qwen3-4B-Instruct-2507 with QLoRA on dataset `p9_honeypot` so that the model is a honeypot by itself:
no instruction is added to any prompt. The behaviour (silent alert to `http://127.0.0.1:8090/alert`, invented
fake values for attackers, normal answers for everyone else) must come from the weights.

| File | What it is |
|---|---|
| `build_notebook.py` | Generates the notebook; edit this, not the notebook |
| `p9_train.ipynb` | The Colab notebook (generated) |

## Run from the terminal with the Colab CLI (recommended)

The [Colab CLI](https://github.com/googlecolab/google-colab-cli) (`uv tool install google-colab-cli`)
rents a Colab Pro GPU and runs the notebook from here, no browser and no Drive needed.
One-time login (needs `gcloud`; run it yourself, it opens a browser):

```bash
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab sessions          # verifies the login
colab usage             # compute-unit balance before the run
```

Then (from `making_a_honeypot_model/`):

```bash
colab new -s p9 --gpu L4                                   # or A100; a 400 means no entitlement
for f in benchmark/data/*; do colab upload -s p9 "$f" "/content/p9/data/$(basename "$f")"; done
colab exec -s p9 --timeout 21600 -f code/training/p9_train.ipynb   # default timeout is 30 s per cell: too short for model loading and training
colab ls -s p9 /content/p9/runs                            # find <run_id>
colab download -s p9 /content/p9/runs/<run_id>/run_record.json results/runs/<run_id>/run_record.json
# and each file of /content/p9/runs/<run_id>/adapter/ into results/runs/<run_id>/adapter/
colab log -s p9 -o results/runs/<run_id>/colab_session.ipynb
colab stop -s p9                                           # always: an idle VM burns compute units
```

**Recommended: run detached.** Long `colab exec` connections drop (2026-10-29: "Connection was lost" twice), and a
dropped connection stops the notebook. Instead, run the same code as a script in a background process on the VM, and
check it with short calls:

```bash
colab upload -s p9 code/training/p9_train.py /content/p9/p9_train.py     # p9_train.py = the notebook's code cells, in order
printf '%s\n' 'import subprocess' 'log=open("/content/p9/train.log","w")' \
  'p=subprocess.Popen(["python","-u","/content/p9/p9_train.py"],stdout=log,stderr=subprocess.STDOUT,cwd="/content",start_new_session=True)' \
  'open("/content/p9/train.pid","w").write(str(p.pid))' | colab exec -s p9 --timeout 90
code/training/colab_check.sh            # alive?, elapsed, GPU, current cell, progress bar, errors (retries dropped connections)
```

The notebook sees `/content/p9/data` and uses it instead of Drive (`USE_DRIVE` is False).
Everything on the VM is lost at `colab stop`, so download first.

## Run in the Colab web UI (Drive)

1. `python3 build_notebook.py` (only after editing the script).
2. Upload the whole `benchmark/data/` folder (all `*.jsonl` and `manifest.json`) to `MyDrive/p9/benchmark/data/`.
3. Open `p9_train.ipynb` in Colab Pro, choose an **L4** (or A100) GPU runtime, then *Runtime > Run all*.
   The notebook stops if any dataset file differs from `manifest.json`, if any example is longer than
   4,096 tokens, or if the loss mask is wrong (system, user and tool-result tokens must not be trained).

Settings live in the first code cell (`TRAIN_CONFIG`: LoRA r 16, alpha 32, dropout 0, all linear projections,
2 epochs, lr 2e-4 cosine, warmup 5%, effective batch 16, seed 20261029, bf16 when supported).

## Outputs (`MyDrive/p9/runs/<run_id>/`)

- `adapter/`: the LoRA adapter
- `merged_16bit/`: the merged model (about 8 GB; switch off with `SAVE_MERGED_16BIT = False`)
- `gguf_*` and `merged_16bit/`: only if `EXPORT_GGUF` / `SAVE_MERGED_16BIT` are set (off since 2026-10-30; the lab serves the adapter with vLLM)
- `requirements.lock.txt`: exact package versions; set `LOCKFILE` to this file to reproduce the run
- `run_record.json`: run id, dataset version and sha256s, model id and revision, training config, library versions,
  loss history, smoke-evaluation results, artefact list

## Use the adapter locally

`P9_RUN=<run_id> code/serve/vllm_ctl.sh start` serves the base model with this run's adapter (see `code/serve/README.md`).
