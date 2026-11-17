# VV-GPT4

A local research studio for **Xavier's own Transformer models**, built directly on VV-GPT3. Train from scratch, continue your existing checkpoints, inspect learning curves and compare learned outputs. No Ollama, pretrained model weights or paid inference service.

## Open

Double-click **VV-GPT4 Preview** on the desktop, or run `VV_GPT4.command`. The studio normally opens at http://127.0.0.1:5054. The launcher reuses this checkout's server or finds a free loopback port. The original **VV-GPT3** launcher, project and checkpoints stay intact.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python launcher.py
```

Installation and the first uncached GPT-2 tokenizer setup need internet. Tokenizer rules are not pretrained model weights. Training/generation then run locally on MPS, CUDA or CPU; the interface has no CDN dependencies.

## Grounded in what you built

VV-GPT3 already had a custom decoder, causal attention, tied embeddings, data preparation, accelerator training, checkpointing, generation and a desktop launcher. Its three existing families include approximately 30M-parameter chat and Shakespeare models and a 51M chat model. This project inherits that code from `XavierB100/vv-gpt3` at `802d1a8` and reads the original `_best.pt` checkpoints without modifying them.

The new architecture is an optional compatible extension to the project, not compatible weights: new revised models use RMSNorm, rotary positions, bias-free projections and an approximately parameter-matched SwiGLU layer. Classic checkpoints retain their original architecture and vocabulary when continued.

## What the latest experiments establish

All training decisions and evaluation limits are recorded in [the protocol](docs/RESEARCH_PROTOCOL.md). [The visual research report](docs/enhancements.html) includes curves, scores, sources and every planned public sample. Open **Research notes** inside the app for the same report.

- **Your Shakespeare checkpoint:** 1,200 additional standard-likelihood updates lowered historical test bits/byte from **2.600 to 2.331** and external Hamlet from **2.518 to 2.338**. Artificial paragraph markers disappeared in the 12 planned outputs. Sentence fragments improved, but grammar and dialogue continuity still fail; lower loss is not proof of reliable coherence. The historical test was part of the original model's validation tail; the external Hamlet excerpt was not used for training or selection.
- **Revised Transformer:** paired Micro runs, same compact vocabulary/data/context and 1,500 steps, two seeds. Full test bits/byte improved **2.775 → 2.624** and **2.759 → 2.606**, about 5.5%. Revised runs used 2.558M parameters versus 2.591M classic. Individual steps were about 10% slower, but reached each classic run's final validation level at **600/700 steps in 28.8/33.3 seconds**, versus classic's 1,500-step runs in 65.5/65.2 seconds. This is discrete, post-hoc time-to-threshold evidence, not an equal-time retraining experiment or a promise for every corpus/model size.
- **Repetition objective:** the bounded unlikelihood adaptation reduced token repetition in Shakespeare, but word repetition was worse than standard continuation. It stays experimental and off by default. Sampling penalties also stay off by default so you can see what the weights learned.
- **Clean-language study:** a revised six-layer Mini model trained from random weights on 20,000 public TinyStories examples, with 500 validation and 500 reserved test stories. See [the retained story results](docs/stories-results.json) and [qualitative review](docs/quality-review.json) for its measured score and all 16 planned continuations. This is a restricted story model, not a general assistant; the different data, task, vocabulary and budget prevent attributing its behaviour to architecture alone.
- **Further story learning:** one declared extra 5,000-step budget with lower dropout reduced prediction loss on a different common reserved slice **0.7528 → 0.6934 bits/byte**. Mean word repetition fell **1.78% → 1.44%** on the same prompts, and some topics stay more consistent. Every reviewed continuation still has entity, goal or logic errors. This is an adaptive exploratory result; see [paired samples](docs/stories-continuation-results.json), including the failures.
- **Existing chat checkpoint:** two conservative continuation candidates changed historical bits/byte from **1.3708 to 1.3697/1.3695** (about 0.1%). This does not establish noticeably better chat. Both started from your actual `chat_best.pt`. [Results](docs/chat-research-results.json) contain aggregate metrics only. Private source text stays local. Historical test selection limits and the lack of independent chat-coherence ratings are explicit in the report.

Earlier negative findings remain in [training-comparison.json](docs/training-comparison.json) and [generation-comparison.json](docs/generation-comparison.json): a smaller vocabulary made steps faster but did not beat the original vocabulary's held-out score at the tested budgets; hard phrase blocking distorted recurring formatting. Different tokenizers require byte-normalized scores on the same text. Scores from different test corpora are not rankings across models.

## The simpler interface

One large learning chart is the main screen. **Experiments**, **Models**, **Data**, **Generate** and **New run** open drawers. A quiet measurement strip, hover values, compatible validation overlays, pace view and collapsed measurement table make the visualizations useful without a dashboard full of panels. Drawers support keyboard focus, Escape/backdrop close and full-width mobile layouts.

To continue one of your trained models, open **Models → Continue training**, choose appropriate training text, name the new run and start. The source weights are copied to that run; architecture, context and vocabulary are inherited. The starting checkpoint is evaluated and saved at step zero. Further training must beat it on validation to replace the best weights. Use a genuinely new holdout when you need independent evidence: automatically splitting text cannot undo a model having seen it previously.

To try the clean-language result, select **Simple stories / further learning** in Generate and begin a story, for example:

> It was raining outside. Mia wanted to stay dry, so she took her

Use your model's learned format for chat or Shakespeare. Generate continues the prompt; it is not an instruction-following chat service. Context is bounded by the checkpoint. Compare another model with the same prompt, seed and sampling settings, or reopen saved samples and their original settings. Unpenalized generation is the default; phrase blocking remains opt-in.

## Training reliability

- Best weights are saved immediately and atomically, separate from the resumable latest checkpoint. Continued runs retain a run-local starting checkpoint.
- Resume restores the model, optimizer, schedule position and random states. Exact interrupted/uninterrupted CPU parity is tested; cross-accelerator determinism is not promised.
- Unicode, emoji, punctuation and speaker labels survive preparation. Common WhatsApp AM/PM formats and multiline messages are handled. Raw decoded uploads are retained; repeated long plain-text paragraphs are removed only from the prepared copy.
- New imports use chronological 80/10/10 train/validation/test partitions. Explicit research partitions are fingerprinted. Tokenizers fit only training text. Fixed train/validation batches use dropout-free evaluation; only validation selects checkpoints.
- Vectorized batch construction, AdamW decay groups, warmup/cosine scheduling, accumulation, early stopping and durable pause/stop/resume are available.
- Revised architecture is the default for new runs, based on the paired study. Original 50,257-token vocabulary and ordinary likelihood remain defaults. Mini adds a modest six-layer, width-192 option. Compact BPE is an explicit experiment.

The app's final test score covers at most the first **8,000 characters** of a run's test partition and is normalized by actual UTF-8 bytes. The research report also has larger, separately labeled evaluations. Do not mix those scores. Near-duplicate text can still contaminate splits; exact deduplication is not semantic deduplication. Repeated speaker labels also count toward token repetition. Small locally trained models can memorize private training text; source and samples remain in local storage.

## Storage and lifecycle

`runtime/` contains SQLite, raw/prepared text, tokenizers, research source data, logs and checkpoints. It is ignored by Git. `VVGPT_DATA` selects isolated storage. Back up the whole directory with app/workers stopped. No automatic background retraining occurs.

Closing the browser/server leaves active training workers running. Reopen the launcher to monitor, pause or stop them. Stopped/interrupted jobs resume from `latest.pt`; a crash may lose steps since the last evaluation save. Finished runs remain fixed. Original model discovery is configured for this Mac in `vvgpt/lab.py`.

The loopback server validates Host/Origin and uses a per-start request token. This is a single-user local development application. The desktop preview may be served by `local.xavier.vvgpt4.preview`; the launcher reuses it. No Ollama job is needed.

Work stays local on `enhance-original-models`; `archive/assistant-alpha` preserves the earlier direction. This project has not been published to GitHub. No new redistribution license has been selected for inherited code. TinyStories text is CDLA-Sharing-1.0; its revision, download bound, partitions and attribution are retained in the research manifest.

## Verification and reproduction

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py launcher.py vvgpt scripts
node --check static/studio.js
bash -n VV_GPT4.command
```

`scripts/verify_ui.cjs` uses Playwright/Chromium and temporary storage to exercise import, actual training, continuation, original-model generation, matched model comparisons, persistence, charts and mobile drawers. Set `PLAYWRIGHT_MODULE` to an installed Playwright package if needed; `PLAYWRIGHT_EXECUTABLE_PATH` can select an existing Chromium binary. `scripts/capture_studio.cjs` captures the real preview after verification.

Research scripts run locally and must be serialized on the accelerator:

- `research_coherence.py`: reconstruct the exact original public preprocessing and run the declared continuation/architecture conditions; reuses completed conditions and resumes interrupted jobs with a saved state. Its source Hamlet HTML must already exist in the local research folder.
- `evaluate_coherence.py`: post-selection public scores and fixed generation prompts; preserves results already collected.
- `prepare_stories.py`, then `train_stories.py`: bounded download of author-published training text, exact whole-story partitioning, train the declared model and retain the planned evaluations.
- `continue_stories.py`: the separately declared 5,000-step exploratory continuation, lower dropout and evaluation on a different reserved test slice; retains the first budget unchanged.
- `research_chat.py`: reconstruct the private original data locally, run conservative continuation conditions and export aggregate scores only.
- `build_report.py`: render retained observations and qualitative review; never trains or samples.
- `audit_legacy.py`: inspect original checkpoints read-only.

Model/data IDs in retained manifests describe this Mac's runtime, not a fresh installation. Experimental timings exclude preflight, tokenizer fitting and model/data initialization; steady step timings also exclude the first ten updates, evaluation and saves. Original Shakespeare continuation trials predate the step-zero safeguard, although both improved validation.

Source layout: `app.py` serves UI/API; `vvgpt/lab.py` manages datasets/workers/checkpoints; `worker.py` trains; `model.py` implements both Transformer blocks; `objectives.py` contains the optional bounded unlikelihood loss; `data.py`, `tokenizer.py` and `evaluation.py` handle preparation and scoring.
