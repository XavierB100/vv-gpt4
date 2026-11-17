# Public examples

`shakespeare.txt` is the public-domain Shakespeare corpus used by the original VV-GPT3 project. It is byte-identical to the upstream [char-rnn tinyshakespeare input](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), verified at release preparation. Source and licensing context: [third-party notices](../THIRD_PARTY_NOTICES.md).

SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

`demo.json` pins the optional public-data model archive, checksum and model-card link. Download happens only when explicitly running `python scripts/install_demo.py`. The archive is a GitHub Release asset, not a tracked Git blob. It contains no original/private chat data or chat-trained weights.
