# Relational Reasoning Probes — Technical Report (v0)

**Status:** working draft. Sections 1 and 2 are first-pass. Sections 3+ are placeholders.

---

## 1. Thesis

We propose that reasoning in transformer language models can be productively
analyzed as motion through a **relational state space** rather than as
next-token prediction in a coordinate-embedded space. Concretely: the
*relational structure* of activations across reasoning steps — the patterns
of co-activation, mutual information, and geometric relations between
activation vectors — is hypothesized to carry the load that underwrites
successful reasoning, more than the absolute coordinate values of those
activations.

This is a claim about transformer internals, not about cognition or
philosophy. It is testable. The first test is a probe on small open-source
language models.

### 1.1 The falsifiable prediction

We pre-register the following prediction before running any experiments:

> A probe trained to recover *relational features* of transformer
> activations (co-activation patterns and pairwise geometric relations
> between activation vectors across reasoning steps) will predict
> reasoning-step success on a held-out reasoning benchmark **at least as
> well as** a probe trained on the absolute activation values, while using
> substantially fewer parameters or substantially less information.

If this prediction is wrong — if the relational probe is meaningfully worse
than the coordinate probe at matched capacity — the central claim of this
program is falsified in its current form and must be revised.

### 1.2 What this is and is not

This program is a mechanistic interpretability research direction. It
proposes a lens on transformer internals and a method for testing whether
the lens reveals load-bearing structure. It is not a new architecture, not
a new training objective, and not a claim about consciousness, cognition,
or general intelligence.

---

## 2. Scope

### 2.1 In scope

- Probing the activations of small, openly-available pretrained transformer
  language models (parameter count below approximately 200M) on
  publicly-available reasoning benchmarks.
- Defining and operationalizing "relational features" of activations in
  ways that an ML researcher can re-implement from the report alone.
- Comparing relational probes against coordinate-value baselines on
  predictive performance for reasoning-step success.
- Pre-registering predictions before experiments and reporting outcomes
  honestly, including negative results.

### 2.2 Out of scope

- Training new models from scratch.
- Architectural modifications to transformers (deferred to a possible
  second milestone, contingent on probe results).
- Claims about physics, cosmology, cognition, or consciousness. This is
  an empirical investigation of transformer internals.
- Comparisons across models larger than approximately 200M parameters
  (compute constraint).

### 2.3 What this is NOT, explicitly

This program is **not** an extension of the Santoro et al. "Relation
Networks" line of work or related relational-reasoning architectures that
add explicit relation-comparison modules to neural networks. Those works
modify the network. We modify nothing in the network — we probe activations
of an existing pretrained model and ask whether a relational lens reveals
structure that a coordinate lens misses.

This program is also **not** a graph neural network proposal, a
neuro-symbolic system, or a knowledge-graph probe in the LAMA / BEAR
tradition. Those are valuable lines of work; they are not what we are
doing.

---

## 3. Background and related work

All citations in this section were verified against primary sources
during Thursday 4 (2026-06-11).

### 3.1 Mechanistic interpretability

This program sits within the mechanistic interpretability tradition:
the project of reverse-engineering what computations a trained neural
network implements, at the level of its internal components.

The foundational methodology we rely on is **probing**. Alain & Bengio
(2016) introduced linear classifier probes: training a simple linear
classifier on the activations of an intermediate layer to measure how
much task-relevant information that layer encodes (arXiv:1610.01644).
The probe's performance is the signal; the probe itself does not modify
the network. This is the core methodology of milestone 1.

A second relevant method is **activation patching** (also called causal
tracing), introduced in its modern form by Meng et al. (2022). Activation
patching works by replacing specific internal activations with cached
activations from a different input, and observing whether the output
changes — establishing causal, not merely correlational, relationships
between internal components and model behavior. We do not use activation
patching in milestone 1, but it is the natural next step if the probe
finds a relational substructure worth investigating causally.

A third relevant line of work is **sparse autoencoders** for decomposing
activation space. Cunningham et al. (2023) showed that sparse
autoencoders trained on transformer residual streams find highly
interpretable features (arXiv:2309.08600). Elhage et al. (2022) provided
the theoretical framing: transformer activations exist in superposition,
with many features compressed into a lower-dimensional space (Transformer
Circuits Thread, 2022). Sparse autoencoders are a tool for decomposing
this superposition. We note this line of work as context; our probe
approach is simpler and does not require training a sparse autoencoder.

### 3.2 Relational reasoning in ML — and how this program differs

The term "relational reasoning" has an established meaning in ML,
originating with Santoro et al. (2017), who proposed Relation Networks
(RNs) as plug-and-play modules that add explicit relation-comparison
computations to neural network architectures (arXiv:1706.01427, NeurIPS
2017). RNs were tested on visual question answering (CLEVR), text-based
question answering (bAbI), and physical reasoning tasks.

**This program is not an extension of the Relation Networks line of
work.** The distinction is fundamental: Santoro et al. ask "can we
make a network reason relationally by modifying its architecture?" We
ask "does an existing pretrained network already represent relational
structure internally, and does that structure carry the reasoning load?"
The former modifies the network. The latter probes it. These are
different questions.

Our program is closer to the mechanistic interpretability tradition
described in Section 3.1 than to the architectural relational reasoning
tradition. The relational framing comes from a theoretical prior about
*which* structure in activations to look for — not from an architectural
design choice.

### 3.3 Biological convergent motivation

Predictive coding (Rao & Ballard, 1999, Nature Neuroscience 2(1):79–87,
DOI:10.1038/4580) offers an independent biological motivation for the
relational-over-coordinate bet. In the Rao & Ballard framework, feedback
connections from higher to lower cortical areas carry top-down
*predictions*, while feedforward connections carry the residual *errors*
between those predictions and actual lower-level activity. What matters
in the network is the error signal — the relational difference between
prediction and observation — not the absolute activation value at any
single neuron.

This is structurally the same bet as the relational probe: that the
*relation* between activations (here, across reasoning steps) carries
more information than the absolute activation values themselves.

We note this as a convergent motivation, not as a load-bearing premise.
The probe results stand or fall on their own empirical merits,
independent of whether predictive coding is the right model of
biological learning.

---

## 4. The program [PLACEHOLDER]

To be drafted Thursday 3 or 4. Will cover:
- Milestone 1: the relational probe (full design, model selection,
  benchmark selection, evaluation protocol).
- Milestone 2 sketch: contingent on milestone 1 outcome.
- Milestone 3 sketch: contingent on milestone 2 outcome.

---

## 5. References [PLACEHOLDER]

To be populated as Section 3 is drafted. Every citation will be verified
against the primary source before inclusion.

---
