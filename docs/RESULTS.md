# Results and their limits

These observations were recorded locally in November 2026. They concern small models and specific text partitions, not general assistant capability. **Lower bits per byte is better only when evaluated on the same text.** Different tokenizers make token loss/perplexity especially unsuitable for direct cross-model comparison.

## Existing Shakespeare checkpoint

| Evaluation | Original best | Standard continued model |
| --- | ---: | ---: |
| Full reserved historical test half, bits/byte | 2.599958 | 2.330834 |
| External Hamlet slice, bits/byte | 2.517907 | 2.337604 |

The continued model starts from Xavier's original checkpoint. Artificial paragraph markers were removed from continued-training text. Grammar and speaker continuity remain weak. The historical test was originally part of the parent's validation tail, so it is **not selection-naive for the parent**. The external Hamlet slice was not used to train/select these runs, but is one slice rather than a broad independent benchmark.

Both standard and unlikelihood continuation conditions are retained in [coherence-results.json](coherence-results.json). Their original runs predate the later protected step-zero safeguard, though both improved validation.

## Revised architecture

| Seed | Classic test bits/byte | Revised test bits/byte |
| --- | ---: | ---: |
| 42 | 2.775195 | 2.623875 |
| 123 | 2.758922 | 2.605716 |

Classic: approximately 2.591M parameters. Revised: approximately 2.558M. The scratch runs use the same data, compact tokenizer, context, batch, 1,500-step budget and declared seed. Revised bundles RMSNorm, RoPE and SwiGLU; these results do not identify which component caused the difference.

Revised steady steps were approximately 10% slower. In a **post-hoc threshold comparison**, the revised runs reached their classic counterparts' final validation loss after 600/700 steps (28.8/33.3 training seconds), compared with classic full-run training time of 65.5/65.2 seconds. This suggests improved learning efficiency in these runs; it is not a new equal-time experiment. Timings exclude preparation/tokenization/initialization, and steady timings exclude the first ten updates, evaluation and saves.

## Public simple-story model

A new 3.443M-parameter revised Transformer trained on 20,000 whole TinyStories records, with 500 validation and 500 test stories. Compact BPE fits only training text. The first budget was 5,000 updates (about 7.9 minutes elapsed on the recorded MPS machine). A declared exploratory continuation added 5,000 updates (about 7.1 minutes), a fresh optimizer and lower dropout.

| Same reserved 100,000-character slice | First budget | Continued budget |
| --- | ---: | ---: |
| Bits/byte | 0.752800 | 0.693430 |
| Observed word repetition across fixed outputs | 1.78% | 1.44% |

The second comparison uses characters 100,000–199,999 of the same test partition, separate from the first evaluation's slice. The continuation decision and slice choice followed earlier inspection; this is an **adaptive exploratory study**, not fresh independent confirmation. Repetition measurements are exploratory, and the same prompts informed further training.

All 16 inspected continued outputs still contained entity, goal or causal inconsistencies. Do not describe the model as a reliable storyteller or conversational assistant. The release demo is this exact continued best checkpoint. The interface shows **0.599866 bits/byte on its smaller 8,000-character app test**; that is a different scope from the 0.693430 comparison above.

Evidence: [first story study](stories-results.json), [continuation study](stories-continuation-results.json), [qualitative review](quality-review.json), [model card](MODEL_CARD.md).

## Private chat and negative findings

The original chat checkpoint's historical test bits/byte changed from **1.370782** to **1.369684 / 1.369457** under two continuation conditions. Roughly 0.1% improvement is not evidence of a meaningful conversational breakthrough. The parent previously used the historical tail for selection. Only [aggregate chat results](chat-research-results.json) are public; source text, output samples and chat-trained weights are excluded.

Earlier compact-vocabulary runs were faster but could reduce quality. Bounded unlikelihood and generation-time repetition controls had mixed results. Penalizing repeated tokens can make numerical repetition smaller while damaging natural repeated speaker labels. All retained conditions matter, including unsuccessful ones.

## How to inspect the evidence

Start with the [research protocol](RESEARCH_PROTOCOL.md), then inspect the JSON result files. [training-comparison.json](training-comparison.json) and [generation-comparison.json](generation-comparison.json) preserve earlier comparisons. [legacy-evidence.json](legacy-evidence.json) records original metadata/checkpoint checks. The in-app Research notes report is generated from retained observations by `scripts/build_report.py`.

A fresh clone contains the evidence and public demo, but not every original checkpoint, source partition or private dataset. Research scripts list those prerequisites in [development notes](DEVELOPMENT.md). Run IDs refer to historical local artifacts. Near-duplicate contamination, limited seeds, short contexts and a small qualitative sample remain important limitations.
