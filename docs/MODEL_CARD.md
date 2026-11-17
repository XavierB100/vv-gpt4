# Model card: VV-GPT simple stories

**Release:** v4.3.0 · **Display name:** Simple stories / further learning · **Author/project:** Xavier Blake / VV-GPT

A small decoder-only language model trained within this project from random initialization, followed by one exploratory continuation. It is intended for inspecting language-model training and simple-story prompt continuation. It is not instruction-tuned and is not a factual assistant.

## Specification

| Property | Value |
| --- | --- |
| Parameters | 3,443,136 |
| Architecture | 6 layers, width 192, 6 attention heads, RMSNorm, RoPE, SwiGLU, bias-free blocks |
| Context | 256 tokens |
| Vocabulary | 4,096-token compact byte-level BPE; fitted only on training stories |
| Training text | 20,000 TinyStories V2 GPT-4 stories; exact whole-record deduplication |
| Reserved data | 500 validation / 500 test stories |
| First budget | 5,000 updates, batch 16, context 256, seed 42, peak LR 0.0005, dropout 0.1 |
| Continuation | 5,000 updates, same batch/context/seed, fresh optimizer, peak LR 0.0003, dropout 0 |
| Selection | Lowest validation likelihood, including protected step-zero source weights |
| Recorded compute | Apple MPS; approximately 7.9 + 7.1 minutes elapsed, excluding preparation |
| Demo run ID | `155f4e2dcc524a9391c072b201364229` |
| Parent run ID | `e65fe480badd4a56ba0011c82815164d` |

## Data provenance

Source: [Ronen Eldan and Yuanzhi Li's TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, file `TinyStoriesV2-GPT4-train.txt`. The bounded download read 17,275,507 bytes to collect 21,000 complete unique stories. Whole stories were split in order and separated with EOS markers. Data is synthetic text published by its authors under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).

No private chat text was used for this model. The release archive contains no source stories and no downloaded foundation-model weights. No blanket license has been selected for these project model weights; see [project notices](../THIRD_PARTY_NOTICES.md). Dataset licensing is documented separately and is not represented as a code/model license.

## Evaluation and known failures

On a shared reserved 100,000-character slice, first-budget versus continued-model bits/byte was **0.752800 → 0.693430**. This is an adaptive exploratory comparison after earlier inspection. The app's separate first-8,000-character score is **0.599866**, so the interface and research headline should not be treated as the same test.

All 16 inspected continued samples still had entity, goal or causal consistency problems. The model may repeat itself, contradict a prompt, change a character's identity, produce abrupt endings or generate inappropriate text inherited from data patterns. It has no factual grounding, instruction tuning or safety fine-tuning. Keep portfolio claims about implementation and measured learning, rather than broad intelligence.

Use the full [results and caveats](RESULTS.md) and [retained public samples](stories-continuation-results.json) to judge behavior. No cherry-picked sample demonstrates general coherence.

## Installation and integrity

```sh
python scripts/install_demo.py
```

The versioned release URL, archive size, archive hash and exact checkpoint hash are in [examples/demo.json](../examples/demo.json). The installer verifies both hashes, validates the model state and imports the original curve as a completed, explicitly marked recorded demo. Repeating installation is idempotent when the same checkpoint exists; conflicting runs are preserved and rejected.

To use a manually downloaded official release archive:

```sh
python scripts/install_demo.py --archive /path/to/vvgpt-stories-demo-v4.3.0.zip
```

Included: `best.pt` (weights/config/tokenizer), `summary.json`, and `record.json` (run config and original measurement rows). Excluded: source text, parent weights, `latest.pt`, optimizer state, private models and private samples. Historical identifiers in the record document provenance, not installed dependencies. The curve's device/time describe its original session.

To learn further, import suitable text and use Models → Continue training. The demo's completed original run is not resumable. A new dataset should provide a fresh independent holdout if the goal is evidence of generalization.
