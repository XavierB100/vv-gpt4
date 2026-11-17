# Using VV-GPT

## Four places to work

**Train** shows the selected experiment and its recorded learning curve. **Generate** tests saved weights. **Models** lists available checkpoints with Generate and Continue training actions. **Data** imports source text. These destinations stay in the top bar; browser Back returns to the previous destination. The selected experiment survives a reload.

Only experiment setup, history and the measurement table open as dialogs. Escape closes a dialog and restores focus. The chart supports Left/Right, Home/End and Escape while focused. Loss and Pace use the same recorded evaluation points; curves are not smoothed.

## First use

A fresh clone starts empty. Use Data → Use Shakespeare for the bundled public example, or run the optional demo installer from the README. Installing the demo gives you a ready model and its historical curve. It does not install a training dataset.

Import UTF-8 text with a `.txt`, `.md` or `.log` extension (up to 16 MiB per request). Raw decoded text is retained separately from the prepared copy. Preparation keeps Unicode, punctuation, emoji and speaker labels. Common WhatsApp exports and multiline messages are recognized. Long exact duplicate paragraphs may be removed from plain-text prepared copies; this is not semantic deduplication.

Use material suitable for the behavior you want: simple stories for story continuation, dialogue for speaker-formatted continuations. A tiny or repetitive dataset can teach patterns while giving very poor generalization.

## Training settings

| Setting | Meaning / practical starting point |
| --- | --- |
| Starting point | Scratch initializes new weights. Continue copies a saved model and inherits its architecture, vocabulary and context. |
| Steps | Optimizer updates. 25 is a wiring check; 1,500 is a first experiment, not a guarantee of coherence. |
| Model size | Micro is the interface default. Nano uses less memory; Mini adds depth. Bigger models need more data and compute. |
| Transformer | Revised uses rotary positions, RMSNorm and gated feed-forward layers. Original retains classic compatibility. |
| Vocabulary | Original GPT-2 vocabulary is the default. Compact BPE is fitted only to training text and can reduce compute, but did not always improve quality. |
| Context | Number of prior tokens visible to the model. Larger context costs memory. Continuation inherits it. |
| Batch / accumulation | Examples per update; accumulation combines smaller batches. Reduce batch first if memory is tight. |
| Learning rate | Peak update scale. Scratch default is 0.0003; the interface suggests 0.0001 for continued learning. These are starting points, not universally optimal values. |
| Evaluate every | Number of updates between train/validation measurements and resumable checkpoints. |
| Stop patience | Evaluation intervals without validation improvement before early stopping. |
| Seed | Controls the run's initial random state. Cross-device exact reproducibility is not promised. |
| Repetition training | Experimental bounded unlikelihood objective. Keep standard likelihood unless running a deliberate comparison. |

Normal imports split chronologically into 80% train, 10% validation and 10% test. Held-out partitions must be long enough for the chosen context. Validation selects the best checkpoint; the final test is evaluated afterwards. Previously seen text does not become an independent test merely by splitting it again.

## Pause, stop and return

One training run owns the training slot at a time. **Pause** keeps the worker and slot alive. **Resume** continues it. **Stop** asks the worker to save and exit; wait for its status to change. A stopped or interrupted run can resume from `latest.pt`. The resume test checks model, optimizer, learning-rate position and RNG continuity on CPU.

Best weights save immediately on validation improvement; latest state saves at evaluation checkpoints. A sudden process kill can lose work after the last saved checkpoint. A finished experiment is fixed: use **Models → Continue training** to make a new experiment. Its starting weights are evaluated and protected at step zero.

Closing the browser or launcher does not stop a training worker. Reopen the studio to monitor it. Stop training explicitly before backing up or shutting down the machine.

## Read results correctly

Training loss measures fit to training text; validation loss estimates performance on its reserved partition. Falling training loss with rising validation loss is a warning of overfitting. Compare curves only with matching data, tokenization and evaluation conditions.

The app's final **test bits/byte** covers at most the first 8,000 characters, using actual UTF-8 bytes. Research reports sometimes use larger slices; do not mix those numbers. The optional demo curve is marked **Recorded demo**, with the original device and timing. It is not a benchmark of your installation.

Choose Compare experiment to overlay validation curves. View measurements gives exact values. The reported step time excludes evaluation/checkpoint overhead; total elapsed is also shown.

## Generate and compare

Use a prompt in the model's learned format. Shakespeare models expect speaker labels and verse; the story demo expects the beginning of a short story. These models continue text rather than obey arbitrary instructions.

The best checkpoint is used. Temperature changes sampling variety; a seed makes comparisons controlled on the same device/software. Top-k 50 and top-p 0.9 are fixed. Repetition penalty defaults to 1 (off), and Phrase guard defaults to off. Suppressing repeated tokens can hide failure or distort legitimate labels; it does not improve the weights.

Enable Compare another result and choose a second model for the same prompt, seed and settings. Leaving the comparison model blank uses the same model without sampling penalties. If penalties are already off, that comparison may be identical. Saved generations restore both output and settings; Export output downloads JSON. Long prompts are trimmed to the model's last context window, with a visible note.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| No models | Install the public demo or finish a training experiment. Original VV-GPT3 discovery is optional. |
| Browser does not open | Open the URL printed by `launcher.py`; try `--no-browser`. |
| Port already in use | The launcher checks up to 20 ports. `VVGPT_PORT=5070 python launcher.py` changes the starting point. |
| Tokenizer download fails | The initial GPT-2 vocabulary cache needs internet access; retry once connectivity is available. Imports use this tokenizer for size checks even when later training uses compact BPE. |
| Model runs out of memory | Reduce batch/context/size for scratch runs. Continued models retain their context and size; reduce batch. |
| Experiment is interrupted | Inspect `runtime/runs/<id>/worker.log`, then Resume if a latest checkpoint exists. |
| Refresh-before-trying message | Server request tokens change on restart. Reload the page. |
| Example is slow on CPU | Use Nano, compact vocabulary and a short budget to verify the workflow. Do not judge language quality from that smoke run. |
| Demo checksum mismatch | Remove only the downloaded archive and fetch the official release again. The installer leaves existing runs untouched. |

## Privacy and backups

Keep private data out of commits, screenshots and shared sample exports. Small models can memorize source text, so private-trained weights may also reveal it. `runtime/`, `local_config.json`, `.venv/` and `artifacts/` are ignored. Stop workers and server, then copy the entire runtime plus local configuration. A source-only GitHub clone does not restore private datasets, generated samples or local checkpoints.
