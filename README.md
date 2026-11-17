# VV-GPT4

A dark, local studio for **Xavier's own Transformer models**, built directly on VV-GPT3. Train from scratch, inspect learning curves, and compare generation from your existing and new checkpoints. No pretrained model weights, Ollama service, or paid inference API is used.

## Open the studio

Double-click **VV-GPT4 Preview** on the desktop, or run `VV_GPT4.command`. The studio normally opens at http://127.0.0.1:5054; the launcher reuses this checkout's server or selects another free loopback port. The original **VV-GPT3** launcher and checkout are unchanged.

For a fresh installation:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python launcher.py
```

Dependency installation and the first uncached GPT-2 tokenizer setup need internet. Tokenizer rules are not pretrained model weights. Subsequent training and generation run locally on MPS, CUDA, or CPU. The interface has no CDN dependencies.

## What carries forward

The decoder, causal attention, learned position embeddings, tied input/output embeddings, size presets, and checkpoint compatibility derive from `XavierB100/vv-gpt3` at `802d1a8`. Your original project already trained approximately 30M-parameter chat and Shakespeare models and a 51M-parameter chat model. This version preserves those checkpoints and reads the original `_best.pt` files for generation.

## What changed

- Simple dark interface with real training/validation curves, hover measurements, training pace, progress, a data table, and mobile layouts.
- Correct best-checkpoint saving. The old final checkpoint could retain later weights while reporting an earlier best score; the new worker saves best weights immediately and atomically.
- Resume restores model, optimizer, learning-rate position, and random state. It reports errors instead of silently restarting. Exact CPU resume is covered by a regression test; identical results across accelerators are not promised.
- Unicode-preserving preparation, common AM/PM WhatsApp formats, multiline messages and speaker labels. Repeated long plain-text paragraphs are removed from the prepared copy; the raw decoded upload is retained.
- Separate chronological 80/10/10 train/validation/test splits. Fixed train and validation evaluation batches run with dropout disabled. Only validation selects the best checkpoint; the best model then receives a final held-out score.
- Vectorized batch construction, AdamW decay groups that exclude bias/norm vectors, warmup/cosine scheduling, accumulation, early stopping, and durable pause/stop/resume.
- Optional compact byte-level BPE (up to 4,096 tokens), learned exclusively from the training split and embedded in checkpoints. The original 50,257-token vocabulary remains the default.
- Seeded, same-checkpoint generation comparisons, an adjustable repetition penalty, optional phrase blocking, saved settings, JSON exports, and a one-model inference cache.

## Measured speed experiment

Three short Nano pilots used the same public Shakespeare source, raw splits, seed 42, batch 8, and 128-token context on this Mac's MPS device. They are experiments, not replacements for your original trained models.

| Vocabulary | Steps | Parameters | Steady training step | Active training | Run elapsed | Test bits/byte ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original, 50,257 | 400 | 8.008M | 52.57 ms | 22.03 s | 26.00 s | **2.708** |
| Compact, 4,096 | 400 | 1.361M | 25.49 ms | 10.65 s | 12.39 s | 2.930 |
| Compact, 4,096 | 800 | 1.361M | 25.84 ms | 21.05 s | 24.16 s | 2.817 |

Compact training was about **2.06× as fast per steady step**, but its test score was worse at both budgets. It therefore stays experimental. The 800-step budget was chosen after observing the first timing, not as a blinded benchmark. This is one seed and one corpus. Timings exclude preflight/tokenizer fitting; steady step timing excludes the first ten steps, evaluation and saving. Run elapsed includes the worker loop, evaluation and saves, but excludes model/data initialization. Compare test scores only on the same raw text; all three used the same first 8,000 bytes of the held-out partition. Different tokenizers give different raw-text context coverage at 128 tokens. Token loss is not comparable across vocabularies.

See [the visual report](docs/enhancements.html), [raw training results](docs/training-comparison.json), and [generation samples](docs/generation-comparison.json). The generation comparison uses the original Shakespeare best checkpoint, three fixed public prompts, 200 tokens and the same seed per pair. A hard four-token phrase guard eliminated repeated four-token phrases by construction but distorted delimiters/speaker labels. The gentler penalty had mixed repetition results. Neither experiment establishes better coherence or factual accuracy; controls change sampling, not learned weights.

## Using your data

Import `.txt`, `.md`, or `.log` text through Datasets, then select it in Training. Use the original vocabulary first and a Nano pilot to verify the data and learning curve. Compact vocabulary is an optional experiment. Larger models need more varied data and training; selecting a larger preset alone does not make output better.

Use the format your model learned in the Playground, such as `Alice: ...\nBob:` for dialogue or `ROMEO:\n` for Shakespeare. The Playground continues the supplied text; multi-turn context must be part of that prompt and is bounded by the model's context window. These models are not instruction-tuned general assistants. Saved comparisons can be reopened with their original settings.

The final test score covers up to the first 8,000 characters of the test split, normalized by actual UTF-8 bytes. It is a useful limited check, not an exhaustive evaluation. Near-duplicate text can still contaminate chronological splits. Repeated format markers also count toward the displayed repetition metric. Training text containing private information can be memorized; samples and data stay in your local runtime.

## Storage and lifecycle

`runtime/` contains SQLite, imported/prepared text, fitted tokenizers, logs and run checkpoints. It is ignored by Git. Set `VVGPT_DATA` to isolate another workspace. Back up the whole directory with app and workers stopped. Older assistant tables are preserved for recovery but are no longer exposed in the interface.

Closing a browser or the web server leaves an active training worker running. Reopen the launcher to monitor, pause or stop it. Stopped/interrupted jobs resume from `latest.pt`; a crash can lose work since the last evaluation save. Finished runs remain fixed. The new UI lists original checkpoints read-only; their discovery path is currently configured for this Mac in `vvgpt/lab.py`.

The loopback server validates Host/Origin and uses a per-start request token. This is a single-user local development application, not an authenticated network service. The desktop preview may already be served by the temporary launchd job `local.xavier.vvgpt4.preview`; the launcher reuses it. No Ollama job is required.

The prior assistant alpha remains on local branch `archive/assistant-alpha`. Development continues on `enhance-original-models`. This project has not been published to GitHub.

## Verification and reproducibility

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py launcher.py vvgpt
node --check static/studio.js
bash -n VV_GPT4.command
```

`scripts/verify_ui.cjs` uses Playwright/Chromium and temporary storage to test import, actual compact training, original-checkpoint generation, saved settings, charts, and mobile views. Set `PLAYWRIGHT_MODULE` to an existing `@playwright/test` package when necessary. It writes `docs/ui-verification.json` and screenshots.

`scripts/compare_training.py` can run new pilots and save a separate results file; it will not replace the retained original evidence. `scripts/compare_generation.py` reproduces the three public-prompt comparisons. `scripts/build_report.py` renders the retained observations into the static visual report. `scripts/audit_legacy.py` inspects the original checkpoints read-only.

Source layout: `app.py` serves the UI/API; `vvgpt/lab.py` manages datasets/workers/loading; `worker.py` trains; `model.py` contains the inherited Transformer; `data.py`, `tokenizer.py` and `evaluation.py` handle preparation, vocabulary and held-out scoring. No new redistribution license has been selected for inherited code.
