# VV-GPT4 direction

Xavier clarified the objective: enhance his own trained Transformer, with more coherent and less repetitive chat/text, shorter training times, and a simple aesthetic dark interface. Downloaded models, Ollama and a generic assistant are outside this direction.

Keep the original decoder, legacy checkpoint compatibility, and train/test workflow. Measure changes on a fixed corpus and retain negative findings. A lower repetition count is not proof of coherent or factual text. Different tokenizers require a byte-normalized evaluation; token losses are not comparable across vocabularies.

Design: dark ink #0b0e14, graphite #111620, raised panel #171d29, text #e7ebf3, training blue #83b9e9, validation lavender #b6a0e9. Native sans for controls and narrative; monospace only for numeric readouts and generated text. A small navigation rail anchors a large training plot with configuration in a quiet adjoining panel. Comparison uses two equally sized sample columns. The curves are the focus, rather than decorative dashboard cards or glow effects.

Review against the brief: remove assistant onboarding, knowledge-library navigation, and generic feature cards. Lead with the user's experiments and an authentic empty plot until metrics exist. Retain one view for datasets, one for training, and one for custom-model generation. Advanced controls stay collapsed. All charts use actual observations, with hover values, accessible summaries, and clear axis units.

The previous assistant alpha remains at Git branch `archive/assistant-alpha`. Existing runtime data is preserved. VV-GPT3 sources/checkpoints are read-only. No automatic background retraining or model downloads.
