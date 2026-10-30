# P9 honeypot lab. Run from code/. Targets are documented in README.md.
#
#   make data       regenerate the dataset and run the validation gate
#   make build      build the Rust lab (release)
#   make test       cargo fmt --check, clippy -D warnings, unit tests
#   make serve      start the local vLLM server (base + honeypot adapter); make serve-stop frees the GPU
#   make check      dataset hashes, collector URL, both models served by vLLM
#   make catcher    standalone request catcher on 8090 (collector) + 9666 (attacker sink), web view on 127.0.0.1:8099
#   make lab        start the collector (127.0.0.1:8090) and the lab API (127.0.0.1:8787)
#   make ui         start the Streamlit lab UI
#   make baseline   eval the base model on every eval file
#   make eval       eval the honeypot model on every eval file
#   make judge      LLM-judge a run (RUN=<run_id>)
#   make report     print a run (RUN=<run_id>), or base vs honeypot (RUN=<hp run> BASE=<base run>)
#   make db         no database in P9 (JSON-lines run folders in results/runs/)

CRATE   := honeypot_lab
BIN     := $(CRATE)/target/release/$(CRATE)
CONFIG  := $(CRATE)/config.toml
LAB     := $(BIN) --config $(CONFIG)
BENCH   := ../benchmark
RESULTS := ../results
FILES   ?= all
LIMIT   ?=

.PHONY: data build test serve serve-stop check catcher lab ui baseline eval judge report db

data:
	cd $(BENCH) && python3 -I generate.py && python3 -I validate.py

build:
	cd $(CRATE) && cargo build --release

test:
	cd $(CRATE) && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test

serve:
	serve/vllm_ctl.sh start

serve-stop:
	serve/vllm_ctl.sh stop

check: build
	$(LAB) check

catcher: build
	cd $(CRATE) && ./target/release/catcher --log ../$(RESULTS)/catcher/requests.jsonl

lab: build
	$(LAB) serve

ui:
	cd lab_ui && uv run streamlit run app.py --server.address 127.0.0.1

baseline: build
	$(LAB) eval --model base --files $(FILES) $(if $(LIMIT),--limit $(LIMIT))

eval: build
	$(LAB) eval --model honeypot --files $(FILES) $(if $(LIMIT),--limit $(LIMIT))

judge: build
	@test -n "$(RUN)" || (echo "usage: make judge RUN=<run_id>"; exit 1)
	$(LAB) judge --run $(RUN)

report: build
	@test -n "$(RUN)" || (echo "usage: make report RUN=<run_id> [BASE=<base run_id>]"; exit 1)
	$(LAB) report --run $(RUN) $(if $(BASE),--compare $(BASE))

db:
	@echo "P9 keeps runs as JSON-lines folders in results/runs/ (no PostgreSQL)"
