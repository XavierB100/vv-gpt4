# VV-GPT4

A local assistant and a lab for your own language models. This is a working **4.0 alpha**, built alongside VV-GPT3. Original data, models, launchers, and Git history are preserved.

## Start

Double-click `VV_GPT4.command`, or the separate **VV-GPT4 Preview** app on the desktop. The launcher opens Ollama when installed, waits for this checkout's server, and opens a browser. It normally uses http://127.0.0.1:5054 and selects another port if needed. Keep Terminal open while using the app.

The assistant uses locally installed Ollama models. This Mac already has six local models; `qwen3:8b` is the initial choice. Cloud entries are excluded and checked again before inference. Nothing downloads models automatically.

For a fresh setup on Apple Silicon:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python launcher.py
```

Install Ollama and a suitable local model separately if they are absent. Dependency installation, model acquisition, and the first uncached GPT-2 tokenizer initialization need internet. Normal chat, indexed knowledge search, and training use local resources after setup. The web interface has no CDN dependencies.

## What works

- Streaming local-model chat, stop control, saved conversations, text export, and recent conversation context.
- Explicit saved preferences across conversations; no hidden automatic memory extraction.
- Local text/Markdown/log document import and SQLite full-text retrieval with inspectable source passages. Follow-up retrieval includes the previous two user questions.
- Scratch training with the original VV-GPT Transformer architecture, GPT-2 BPE, Apple MPS/CUDA/CPU selection, gradient accumulation, a fixed validation sample, early stopping, pause, stop, and checkpoint resume.
- An independent worker per training job, SQLite run/metric history, atomic best/latest saves, dataset fingerprints, optimizer and RNG restoration.
- Read-only discovery of VV-GPT3 metadata and **best** checkpoints. Generation uses restricted `weights_only` loading with only the legacy config dataclass allowlisted.
- A playground for the original and new custom models. These are text-continuation models, distinct from the instruction-following assistant.

## Data and controls

`runtime/` holds the local SQLite database, imported training text, run logs, and checkpoints. It is ignored by Git. Set `VVGPT_DATA` to a different directory before starting to isolate an installation. Back up the complete directory with the app and workers stopped; SQLite may have WAL companion files while running.

The server is loopback-only, validates Host and Origin, and requires a random per-start request token for mutations. It is a single-user local app, not an authenticated network service. The Flask development server is intentional for this alpha; do not expose it to a network.

Closing the browser does not stop training. Closing the web server also leaves a training worker running. Reopen the launcher to monitor, pause, or stop it. A stopped run can resume its latest saved state. A completed or early-stopped run stays immutable; create a new experiment to continue research. A process crash can lose work since the last saved checkpoint. Cross-device numerical determinism is not promised.

Removing a knowledge file removes it from future retrieval. Previously saved conversations retain any source excerpts they used; delete those conversations separately. Clearing saved preferences affects future prompts, not existing messages. Data is stored in plaintext under your macOS account.

## Current limits

- Document search is lexical, not embedding-based. No PDF parser yet. Citations identify supplied passages; answer correctness is not automatically verified.
- Recent chat context is bounded by character budgets, not an unlimited memory or a model-specific token accountant. Long or token-dense input can reach the engine's context limit. Interrupted/truncated assistant replies are displayed but not reused as complete exchanges.
- Stop takes effect on the next inference output/read; an unresponsive model can hold the slot until the local request timeout.
- New training consumes raw UTF-8 text. Legacy WhatsApp timestamp stripping is not ported; clean exports before new training. Training split is the last 10% of tokens, and repeated/near-duplicate documents can contaminate validation. Loss here is not a benchmark of instruction following or general intelligence.
- No adapter fine-tuning, vision, voice, autonomous tools, web browsing, model downloads, or paid APIs in this release.
- No migrations or automatic updater yet. Original model discovery currently defaults to this Mac's verified VV-GPT3 path in `vvgpt/lab.py`.

## Architecture

`app.py` serves the offline UI and API. `vvgpt/store.py` owns SQLite; `knowledge.py` builds bounded context; `providers.py` talks only to loopback Ollama; `lab.py` manages datasets, model discovery and workers; `worker.py` runs training; `model.py` carries the original custom Transformer forward. `static/` contains plain JavaScript and CSS, without a frontend build chain.

The assistant and scratch models are separate engines. Importing a document for retrieval does not update any model's weights. This separation preserves the original learning project while providing useful pretrained intelligence today. The next bridge is measured adapter fine-tuning, not relabelling scratch training as a general assistant.

## Verify

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py launcher.py vvgpt
node --check static/studio.js
bash -n VV_GPT4.command
```

Optional real-model checks: `scripts/verify_live.py` uses installed `qwen3:8b`, performs a tiny six-step training job in temporary storage, and loads the original Shakespeare **best** checkpoint read-only. It writes `docs/live-verification.json`. `scripts/audit_legacy.py` reproduces the checkpoint comparison without modifying the source project and writes `docs/legacy-evidence.json`.

`scripts/verify_ui.cjs` exercises real chat, source display, persistence, dataset upload and six-step training through a temporary local server, then checks desktop/mobile layouts. It requires `@playwright/test` with Chromium; set `PLAYWRIGHT_MODULE` to an existing package installation if necessary. It writes `docs/ui-verification.json` and screenshots. These optional tests use temporary workspaces, not the user's conversations or training data.

See `docs/deep-dive.html` for the original-project audit and successor roadmap. `docs/DESIGN.md` records the interface direction. Screenshots in `docs/` show verified layouts, not a hosted deployment.

## Provenance

Custom Transformer code derives from XavierB100/vv-gpt3 commit `802d1a8`. The original project was inspected locally and on GitHub. This project is local and has not been published to GitHub. No new redistribution license has been chosen for the inherited code.
