# VV-GPT4 direction

Enhance Xavier's own trained Transformer: coherent, less repetitive text and chat; better learning efficiency; a simple, aesthetic dark research interface. Original VV-GPT3 source, checkpoints and launcher stay intact. The earlier assistant alpha remains on `archive/assistant-alpha`.

The current design responds to the user's further clarification that layout and navigation felt awkward. Keep one large learning chart as the training home, with a persistent Train / Generate / Models / Data navigation bar. Core tasks use full workspace views; experiment switching, setup and raw measurements use focused drawers. Browser Back, selected-run persistence, searchable libraries and a contextual Try this model action make the navigation predictable. See INTERFACE_REVISION.md for the design rationale.

Palette: graphite #131416, workspace #1b1d20, raised controls #24272b, borders #35393f, text #f0f1f3, secondary #acb3bd. Training periwinkle #a7b4fa and validation teal #7bc6b5. Locally hosted Manrope supplies interface text and tabular numbers, with system sans fallback. The chart is one composed surface; generation uses a focused input/output split, and libraries use compact rows. Advanced settings remain collapsed.

Improve learned weights and evaluate them separately from sampling tricks. Compare matched data/configuration and preserve losing candidates. Different tokenizers require byte-normalized scores on the same text. Likelihood and repeated phrases do not measure semantic coherence. Show full public samples and state domain/selection limitations.

New runs default to the revised block following the two-seed Micro comparison; original vocabulary and ordinary likelihood remain defaults. Existing checkpoints retain their own architecture, context and vocabulary when continued. Step-zero best protection prevents a worse continuation from replacing the starting weights. Original models are available for matched generation comparisons. Unlikelihood and phrase blocking remain optional experiments.

Public clean-language text may be used to test learnable behaviour without pretrained weights. No automatic background retraining, paid services, private-text upload or new pretrained-model downloads. No general assistant claim from a small story model.
