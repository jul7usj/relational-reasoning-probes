# Future directions

The long-term motivation behind this program is a research direction
toward AI systems whose reasoning is organized around relational
structure rather than coordinate-based representation. A natural extension
is to compare mechanisms identified by relational probing in transformers
against what is known about relational reasoning in biological neural
computation.

**This is not in scope for the current program.** It is logged here so
that future iterations of the program have a clear pointer forward.
Revisit only after milestone 1 ships with a real result.

---

## Predictive coding connection

A YouTube summary of Artem Kirsanov's video "The Brain's Learning Algorithm
Isn't Backpropagation" surfaced an independent convergent motivation for
the relational probe angle.

Predictive coding (Rao & Ballard, 1999 — citation to be verified) frames
biological learning as energy minimization driven by prediction error
signals between layers. The key insight: what matters in the network is
not the raw activation value at each neuron, but the *relational signal*
— the difference between a top-down prediction and a bottom-up reality.
Error neurons compute this difference explicitly. The whole framework says
relations between activations, not absolute values, carry the load.

This is structurally the same bet as the relational probe: that relational
structure between transformer activations carries more load than absolute
coordinate values.

**This connection belongs in Section 3 (background and related work) of
the technical report, as a biological convergent motivation.** It is not
a load-bearing premise — the probe stands or falls on its own empirical
results. But it is a real independent line of thinking pointing the same
direction, and it strengthens the motivation section.

Action item: when drafting Section 3, verify Rao & Ballard 1999 as the
foundational predictive coding citation, and find 1–2 papers connecting
predictive coding to transformer internals if they exist.