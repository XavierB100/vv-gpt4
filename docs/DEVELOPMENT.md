# Development, verification and reproduction

## Environment

Use Python 3.11 and the pinned `requirements.lock`. `requirements.txt` describes the supported direct dependencies. Node 20+ is optional and is used only for browser verification; the product has no Node runtime or bundler. macOS Apple Silicon and Linux CPU are release check targets. CUDA is selected when available but has not been separately benchmarked; Windows support is not verified.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q
python -m compileall -q app.py launcher.py vvgpt scripts
node --check static/studio.js
bash -n VV_GPT4.command
git diff --check
```

Tests cover data preparation, train-only tokenization, model/sampling behavior, exact CPU resume, protected continuation, request boundaries, saved sample settings, portable example import and checksum-verified demo installation. These checks support correctness of the workflow; they do not certify model intelligence.

## Real browser workflow

```sh
npm ci
npx playwright install chromium
VVGPT_PYTHON=python npm test
```

GitHub CI uses the Chrome executable already included in the Ubuntu 24.04 runner, avoiding an extra browser/OS dependency installation. For a Linux machine without browser libraries, use `npx playwright install --with-deps chromium`. If Chromium is already installed, `PLAYWRIGHT_EXECUTABLE_PATH` selects its executable. `PLAYWRIGHT_MODULE` can select an existing Playwright module; otherwise the pinned local package is used.

`scripts/verify_ui.cjs` starts a server at port 5064 with an empty temporary workspace and an absent legacy-model directory. It imports text, really trains a short compact model, generates from those weights, restores saved samples/settings, continues a copy of the model and compares the checkpoints. It checks chart inspection, persistent destinations and mobile overflow. It never needs private training data or original checkpoints. Do not run a second copy on the same port.

On success the temporary workspace is removed; on failure it remains for inspection. The server is stopped at the end. If a training-time failure interrupts the test itself, inspect/stop any surviving worker before deleting its temporary files. `docs/ui-verification.json` is the latest recorded check, not a substitute for rerunning it after changes.

## Screenshots

`scripts/capture_studio.cjs` captures the real running studio. Set `STUDIO_URL` to its loopback address; `--generate` creates a sample with the documented public prompt/seed. It expects the story demo's display name. The portfolio captures use a separate workspace containing only the public demo and bundled example. They contain real recorded results, not simulated chart data.

`scripts/check_navigation.cjs` and `check_research_ui.cjs` are additional checks against the original research workspace and expect its named runs. They are not portable fresh-clone CI tests. `build_interface_preview.py` builds a screenshot viewer; `build_report.py` builds the self-contained research report from retained evidence.

## Research scripts

Run accelerator workloads serially. Research scripts are explicit command-line actions; the app never launches them automatically.

| Script | Prerequisites / purpose |
| --- | --- |
| `audit_legacy.py` | Original metadata/checkpoints available through configured legacy path; inspect read-only. |
| `research_coherence.py` | Original VV-GPT3 source, original Shakespeare checkpoint, local Hamlet source HTML and existing Shakespeare import; declared continuation/architecture conditions. |
| `evaluate_coherence.py` | Completed coherence research manifest and checkpoints; post-selection scores and fixed prompts. |
| `prepare_stories.py` | Internet access; bounded download of pinned author-published text and explicit whole-story partitioning. |
| `train_stories.py` | Prepared stories manifest; first declared scratch budget. |
| `continue_stories.py` | Completed first story run and source data; declared exploratory second budget. |
| `research_chat.py` | Original private source/checkpoint available locally; aggregate export only. Do not publish source, outputs or private-trained weights. |
| `compare_training.py`, `compare_generation.py` | Earlier original-runtime experiments; retained for provenance, not turnkey demo commands. |
| `build_report.py` | Existing result JSON in docs; renders evidence without training/sampling. |

The historical research drivers use this checkout's `runtime/` and exact historical IDs/manifests. `VVGPT_DATA` configures the application/installer; it does not automatically redirect every historical research script. Use a separate checkout and inspect script prerequisites before reproducing a study. Reproducing every historical result requires artifacts deliberately absent from GitHub.

For the public story study, prepare data with `python scripts/prepare_stories.py`, then run `python scripts/train_stories.py`; the continuation is `python scripts/continue_stories.py`. Inspect each script and the protocol first. Downloads/compute occur only when explicitly run. A matching seed on a different device/version does not guarantee identical weights or timing.

## Release discipline

Before publishing: stop training; run tests and the fresh-workspace browser check; inspect screenshots and markdown links; audit tracked files and reachable history for private material; verify source lineage. Keep runtime and machine-local configuration out of Git. Publish only the public-data demo archive whose hashes match `examples/demo.json`.

The GitHub Actions workflow checks Python on macOS and Linux, syntax/shell checks, and a real CPU browser workflow on Linux. The repository's Actions tab is the authoritative status for the published revision.
