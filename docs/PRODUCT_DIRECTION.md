# VV-GPT4 direction

Enhance Xavier's own trained Transformer: coherent, less repetitive text and chat; better learning efficiency; a simple, aesthetic dark research interface. Original VV-GPT3 source, checkpoints and launcher stay intact. The earlier assistant alpha remains on `archive/assistant-alpha`.

The selected design is **research first: one large learning chart, everything else in drawers**. No permanent sidebar, card grid, hero or always-open settings column. A quiet top bar opens Experiments, Models, Data, Generate and New run. The chart carries real measurements, with hover values, compatible validation overlays and a pace view. Below it, one compact measurement strip and a collapsed accessible table. Native modal dialogs provide focus handling, Escape/backdrop close and full-width mobile drawers.

Palette: midnight #080a0f, surface #10131b, hairline #252936, text #e4e7ed, secondary #959dad, training blue #91b8f3, validation lilac #bea6e7. Native sans; normal-case labels, restrained borders, generous space. Generated text uses readable prose typography. Advanced training and sampling controls stay collapsed.

Improve learned weights and evaluate them separately from sampling tricks. Compare matched data/configuration and preserve losing candidates. Different tokenizers require byte-normalized scores on the same text. Likelihood and repeated phrases do not measure semantic coherence. Show full public samples and state domain/selection limitations.

New runs default to the revised block following the two-seed Micro comparison; original vocabulary and ordinary likelihood remain defaults. Existing checkpoints retain their own architecture, context and vocabulary when continued. Step-zero best protection prevents a worse continuation from replacing the starting weights. Original models are available for matched generation comparisons. Unlikelihood and phrase blocking remain optional experiments.

Public clean-language text may be used to test learnable behaviour without pretrained weights. No automatic background retraining, paid services, private-text upload or new pretrained-model downloads. No general assistant claim from a small story model.
