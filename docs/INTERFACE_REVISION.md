# Interface revision

The user rejected the previous interface and clarified that layout and navigation feel awkward. Preserve dark simplicity and a chart-focused training home, but give the core tasks stable destinations.

## Design plan

Palette: canvas #131416, workspace #1b1d20, raised controls #24272b, borders #35393f, text #f0f1f3, secondary #acb3bd. Curves use periwinkle #a7b4fa and soft teal #7bc6b5. No glow or decorative gradients; a faint chart fill encodes the validation series only.

Type: locally hosted Manrope for interface, headings and numbers, with system sans fallback. 14px control text, 16px body, 30px run heading. Medium weights and tabular numerals. Labels must remain readable on mobile.

Layout: a persistent four-destination navigation bar keeps Train, Generate, Models and Data visible. Train has a run switcher, run heading, one composed chart surface and a compact outcome footer. Setup and experiment selection are modal sheets; generation and libraries are ordinary pages. All content aligns to one centered maximum-width workspace.

    vv.gpt          Train  Generate  Models  Data          Local
    Experiments / selected run                        New run
    Run title                                Generate from run
    ┌ Learning curve ────────────── Loss | Pace ───────────────┐
    │ Best validation    Steps    Elapsed    Step time        │
    │                                                        │
    │                  Recorded learning curve               │
    │                                                        │
    └ Legend ─── recorded measurements ────────── comparison ─┘
    Model context                         Held-out test detail

Review against brief: a permanent sidebar would take space from the chart and reintroduce a rejected layout. Repeated modal views made the core workflow awkward, so main navigation now changes pages inside the same shell. A single chart surface supplies structure without a dashboard of cards. Navigation uses words and persistent active states. Larger text, shorter travel between controls and a focused generation workspace carry the redesign; decoration stays quiet.

Behaviour: selected experiments persist locally, page destinations use URL hashes, browser back restores the prior page, setup sheets restore focus to the invoking control, and Generate from run selects the exact checkpoint. Search narrows long model/dataset/experiment lists. Finished runs do not show disabled training controls. Mobile navigation stays visible, and layouts collapse without horizontal scrolling. All plots retain unsmoothed observations and disclose their score scope.
