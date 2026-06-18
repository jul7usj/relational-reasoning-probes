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

## 4. The program

### 4.1 Milestone 1 — the relational probe

**Status:** design complete. Implementation pending (blocked on
TransformerLens install — requires fast internet connection).

#### 4.1.1 Setup

- **Model:** GPT-2 small (124M parameters), loaded via TransformerLens
  (`HookedTransformer.from_pretrained("gpt2")`). Weights sourced from
  HuggingFace. No fine-tuning — we probe the pretrained model as-is.
- **Benchmark:** bAbI tasks 1–3 (Weston et al., 2015), loaded via
  HuggingFace datasets (`facebook/babi_qa`). Tasks 1–3 cover one-hop,
  two-hop, and three-hop reasoning respectively. Each problem consists
  of context sentences, a question, and an answer. Supporting facts
  (the reasoning steps necessary to answer) are explicitly labeled.
- **Hardware:** CPU-only inference on a 12GB RAM laptop. Batch size
  10–50 problems. Expected memory usage well under 12GB for GPT-2
  small activation extraction. [to be confirmed on first run]

#### 4.1.2 What we extract

For each problem in bAbI tasks 1–3, we run GPT-2 small on the context
sentences and extract activations at every layer using TransformerLens's
hook points. Specifically, we extract the residual stream activations
at each layer boundary — the vectors that represent the model's
accumulated state after processing each token.

For a problem with N supporting facts (N = 1, 2, or 3 depending on
the task), we extract the activation vectors at the positions
corresponding to the final token of each supporting fact sentence.
This gives us N activation vectors per problem, one per reasoning step.

#### 4.1.3 Relational features vs. coordinate features

We define two families of features for each problem:

**Coordinate features (baseline):** the raw activation vectors
themselves — their absolute values at each layer. A coordinate probe
is trained to predict reasoning-step success from these raw values.

**Relational features (hypothesis):** pairwise relations between
the N activation vectors — specifically:
- Cosine similarity between each pair of activation vectors
- L2 distance between each pair
- Element-wise difference vectors between each pair

A relational probe is trained to predict reasoning-step success from
these pairwise relational features, without access to the raw
coordinate values.

The key design constraint: **the relational probe uses strictly less
information than the coordinate probe.** If the relational probe
matches or exceeds the coordinate probe's predictive performance, the
central claim is supported. If it falls meaningfully short, the claim
is falsified in its current form.

#### 4.1.4 What "reasoning-step success" means operationally

For each problem, GPT-2 small either produces the correct answer or
not. We define reasoning-step success at the level of the full
problem: a problem is "succeeded" if the model's top-1 prediction
for the answer token matches the ground truth answer. This gives a
binary label per problem.

We split problems into succeeded and failed, then train probes to
predict this binary label from the activation features defined above.
Probe architecture: logistic regression (linear probe), consistent
with the Alain & Bengio (2016) methodology.

#### 4.1.5 Pre-registered prediction

Stated before any probe is trained, per the commitment in Section 1.1:

> A logistic regression probe trained on relational features
> (pairwise cosine similarities, L2 distances, and difference vectors
> between activation vectors at reasoning-step positions) will achieve
> held-out accuracy on the succeeded/failed binary label that is
> **within 5 percentage points** of a logistic regression probe
> trained on the raw coordinate activation vectors, using a
> held-out test split of 20% of the bAbI tasks 1–3 problems.

The 5 percentage point threshold is the operationalization of
"at least as well as" from Section 1.1. If the relational probe
falls more than 5 points below the coordinate probe, the prediction
is falsified.

#### 4.1.6 Evaluation protocol

- Train/test split: 80/20, stratified by task (1, 2, 3) and outcome
  (succeeded/failed).
- Probe: logistic regression, scikit-learn default hyperparameters.
  No hyperparameter tuning — we use defaults to avoid overfitting the
  probe to the data.
- Metric: held-out accuracy (primary), ROC-AUC (secondary).
- Reporting: both relational and coordinate probe results reported
  regardless of outcome. Negative results reported honestly.

### 4.2 Milestone 2 sketch (contingent on milestone 1)

If the relational probe succeeds (relational features within 5pp of
coordinate features): investigate *which layers* carry the relational
signal most strongly. Use TransformerLens activation patching to test
whether the relational substructure is causally load-bearing, not
merely correlational.

If the relational probe fails: diagnose why. Candidate explanations:
(a) GPT-2 small does not represent relational structure at bAbI
scale, (b) our operationalization of "relational features" is wrong,
(c) the claim is false. Each leads to a different next step.

### 4.3 Milestone 3 sketch (contingent on milestone 2)

If milestone 2 finds a causally load-bearing relational substructure:
propose a minimal architectural modification that makes this structure
explicit. Test on a small trained-from-scratch model at toy scale.
This is the architectural proposal deferred from Section 2.2.

*Milestones 2 and 3 are contingent. Milestone 1 is the only committed
deliverable of the current program.*

---

*Section 4 design finalized: Thursday 5 (2026-06-11).*
---

## 5. References [PLACEHOLDER]

To be populated as Section 3 is drafted. Every citation will be verified
against the primary source before inclusion.

---
