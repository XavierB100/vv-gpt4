# Saved milestone and resume handoff

**Milestone:** v4.3.0 · **Date:** 17 November 2026 · **State:** portfolio release; feature/research work paused.

The accepted interface has persistent Train / Generate / Models / Data destinations. Keep that design as the starting point. The project centers Xavier's own model architecture and trained weights. No additional Ollama/provider integration or broad redesign is needed for this milestone.

## Saved publicly

- Source, regression/browser checks and pinned dependencies.
- README, portfolio case study, architecture, user guide, model card, evidence and real screenshots.
- Retained public research samples and aggregate private-chat measurements.
- A separately downloadable public story checkpoint with exact hashes and original learning curve.

The GitHub repository is [XavierB100/vv-gpt4](https://github.com/XavierB100/vv-gpt4); the downloadable milestone is [v4.3.0](https://github.com/XavierB100/vv-gpt4/releases/tag/v4.3.0). Main is the released source line. Historical development branches are retained locally; their ancestor commits remain in the published main history. Runtime storage is excluded.

## Saved locally

The working runtime contains original research source artifacts, trained checkpoints, private data and saved generations. These stay outside GitHub. The original VV-GPT3 checkout and desktop launcher remain separate. `local_config.json` preserves this installation's optional original-model discovery.

A local backup should include runtime, machine-local configuration and launcher wiring with no training active. The release closeout records the actual backup path and verification outcome in ignored `runtime/release-handoff.json`; this avoids embedding machine/private storage details in public documentation.

## Verified scope

Python regression checks cover preparation, tokenization, model logic, continuation, CPU resume, local API boundaries and demo installation. Browser checks use a fresh isolated workspace and perform actual import, train, generate, save/reload, continue and comparison operations. The public demo is exercised separately and screenshots are visually reviewed. Published CI status is available in GitHub Actions; local success and remote status are distinct evidence.

Model quality is limited. Story outputs still fail consistency; chat continuation did not show meaningful progress; likelihood is not a semantic coherence test. Read [RESULTS.md](RESULTS.md) before adding portfolio claims.

## If work resumes

1. Preserve the current released baseline and local runtime backup.
2. Define fresh, independent coherence prompts and scoring criteria before tuning.
3. Compare one small data/training intervention at a time with a fixed compute budget and matched evaluation.
4. Retain all outputs and negative results. Do not optimize repeatedly against the existing showcase prompts.

A licensing decision, broader hardware verification and a more systematic accessibility audit remain possible maintenance tasks. They are not blockers to documenting this personal portfolio project. No new training or feature expansion is scheduled.
