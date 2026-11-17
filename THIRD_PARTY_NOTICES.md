# Provenance, attribution and license status

## Project code and weights

VV-GPT4 continues Xavier Blake's [VV-GPT3](https://github.com/XavierB100/vv-gpt3), with earlier public lineage through homemade-gpt, vv-gpt-mac and vv-gpt2. The audited original source was commit `802d1a8`. The model implementation retains original-compatible structures alongside the revised decoder. Public repository history documents the development sequence.

No blanket redistribution license has been selected for project code or the optional model weights. This repository's public availability is not a license grant. Do not infer that third-party data/font licenses apply to all code or checkpoints. Contact the repository owner about reuse. The code and research revision used AI assistance, as described in the case study.

## Bundled Shakespeare example

`examples/shakespeare.txt` is byte-identical to [Andrej Karpathy's char-rnn tinyshakespeare input.txt](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), verified when preparing this release. Shakespeare's original writings are public-domain texts; attribution is retained to the dataset source. The upstream [char-rnn README](https://github.com/karpathy/char-rnn/blob/master/Readme.md#license) states “MIT”; no standalone license file was present when inspected. This bundle contains the public-domain text, not char-rnn implementation code.

SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` · 1,115,394 bytes.

The same corpus was already present in VV-GPT3. Including it makes the example independent of a private/local checkout.

## TinyStories

Author-published dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), by Ronen Eldan and Yuanzhi Li. Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Dataset terms: [Community Data License Agreement – Sharing 1.0](https://cdla.dev/sharing-1-0/). A copy is retained in [docs/licenses/CDLA-Sharing-1.0.txt](docs/licenses/CDLA-Sharing-1.0.txt).

The research used a bounded prefix of `TinyStoriesV2-GPT4-train.txt`; counts, partition hashes and provenance are in the result manifests and model card. Training text is downloaded only by the explicit research preparation script and remains in ignored local runtime storage. The optional demo archive contains resulting weights and recorded metadata, not a distribution of the full dataset.

## Font and dependencies

Manrope is bundled locally under the SIL Open Font License; see [static/fonts/OFL-Manrope.txt](static/fonts/OFL-Manrope.txt).

Flask, PyTorch, NumPy, tiktoken and Hugging Face tokenizers are installed from their upstream distributions; their licenses accompany those distributions. Pytest and Playwright are development/test tools. Version pins appear in `requirements.lock` and `package-lock.json`. The tokenizer vocabulary is not pretrained language-model weights.

## Research references

Architecture and training ideas are credited in [the research protocol](docs/RESEARCH_PROTOCOL.md): LLaMA, RMSNorm, RoPE, SwiGLU, unlikelihood training, compute-optimal training and TinyStories. These papers motivated local experiments; their larger-scale results are not claimed as achievements of this project.
