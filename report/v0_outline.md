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

## 3. Background and related work [PLACEHOLDER — draft Thursday 4]

This section situates the program in the existing ML literature.
All citations will be verified against primary sources before inclusion.

### 3.1 Mechanistic interpretability

The program sits within the mechanistic interpretability tradition:
understanding what computations a trained neural network implements,
at the level of individual components. Key prior work to cover:

- Probing methods: linear probes on transformer activations to recover
  linguistic and factual structure. [citations to be verified]
- Activation patching: causal intervention methods for identifying
  which components mediate specific behaviors. [citations to be verified]
- Sparse autoencoders and superposition: recent Anthropic work on
  decomposing activation space into interpretable features.
  [citations to be verified]

### 3.2 Relational reasoning in ML

Prior work that uses "relational" framing — and how the present program
differs:

- Santoro et al. (2017), Relation Networks: adds explicit
  relation-comparison modules to neural networks. We modify nothing
  in the network — we probe existing activations.
- ReCogLab (DeepMind, 2024): framework for testing relational reasoning
  in LLMs. Related but different methodology.
- Key distinction: prior work asks "can we make models reason
  relationally?" We ask "do existing models already represent relational
  structure internally, and does it matter?"

### 3.3 Biological convergent motivation

Predictive coding (Rao & Ballard, 1999 — to be verified) independently
motivates the relational-over-coordinate bet: in predictive coding
frameworks, what matters is the prediction error signal (a relation
between layers), not the absolute activation value. This is a convergent
motivation, not a load-bearing premise.

*Full draft of Section 3 scheduled for Thursday 4.*
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
