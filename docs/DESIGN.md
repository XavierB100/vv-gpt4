# VV-GPT4 interface direction

The subject is Xavier's personal machine-learning studio: a calm place for useful conversations and visible experiments, with an obvious distinction between an assistant and a model trained from scratch.

- Palette: paper blue `#f8f9fc`, sidebar mist `#edf1f8`, ink `#26334b`, working indigo `#4e63a7`, validation mauve `#ae839f`, local-status green `#438575`.
- Type: native macOS system sans for familiar desktop controls, restrained 32–45px display text, 12–14px working text. Monospace is reserved for generated text in the custom-model playground.
- Layout: stable left navigation; assistant has a central conversation and anchored composer; training has configuration beside evidence; models have named cards with comparable metadata. On narrow screens navigation moves above the content and saved conversations have a selector.
- Distinctive element: a quiet orbital VV mark ties the assistant to the original experimental identity. No animated background, stock photography, CDN fonts, or decorative dashboard metrics.

The first design review rejected a dense generic dashboard for the assistant landing screen: the useful action is a conversation, so there is no row of unrelated metrics. Numeric telemetry belongs in the training lab where it helps the user decide whether learning is improving.

Implementation review: desktop and 390px layouts were inspected using actual screenshots; keyboard focus, label associations, reduced-motion handling, output escaping and error/empty states are included. The small-screen review led to adding a saved-conversation selector, because hiding the sidebar history would otherwise remove access to a core feature.
