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

> **STATUS AS OF THURSDAY 13 (2026-08-20): THIS PREDICTION IS
> FALSIFIED.** See Section 4.7. Under controls equalizing information
> access between probes, relational features show no advantage over
> coordinate features on any task, and a significant *disadvantage*
> on Task 2. The rotation-invariant components of the relational
> feature set (cosine similarity, L2 distance, and the full Gram
> matrix) carry real but substantially weaker signal than the
> basis-dependent representation. Sections 4.4 and 4.5 below record
> earlier results that did not survive these controls and should be
> read together with Section 4.7.

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

### 4.4 Milestone 1 results (Thursday 9, 2026-07-30)
> **⚠ SUPERSEDED — READ SECTION 4.7 FIRST.** The result below was
> produced with an uncontrolled comparison: the relational probe had
> access to the question activation vector while the coordinate probe
> did not. Section 4.7 shows this confound accounts for the effect.
> The parity reported here is retained for the record, not as a
> supported finding.

**Configuration:** GPT-2 small (124M), bAbI Task 1, layer 11 (final
layer), 500 problems, 80/20 train/test split, logistic regression
probes with StandardScaler, random seed 42.

**Label definition (revised from Section 4.1.4):** above-median log
probability of the correct answer token, given the full story +
question as prompt. Revision reason: GPT-2 small top-1 exact match
accuracy on bAbI Task 1 zero-shot = 0.0% — the model never predicts
the bare answer word as its top-1 next token. Log probability captures
graded reasoning quality even when exact match fails. Median split
produces balanced labels (250 high / 250 low).

**Results:**

| Probe | Accuracy | AUC |
|---|---|---|
| Coordinate (768-dim raw activations) | 69.0% | 0.767 |
| Relational (cosine sim + L2 dist + diff vector, 770-dim) | 69.0% | 0.770 |

Gap (coordinate − relational): 0.0 percentage points.
Both probes substantially above chance (50%).

**Pre-registered prediction verdict:** SUPPORTED.
Relational probe accuracy (69.0%) is within 5 percentage points of
coordinate probe accuracy (69.0%). The relational structure of
activations captures the same predictive signal as the full coordinate
representation.

**Interpretation:** The relational features — pairwise cosine
similarity, L2 distance, and difference vector between the supporting
sentence activation and the question context activation — contain
exactly as much information about GPT-2's reasoning quality as the raw
768-dimensional activation values. The coordinate representation adds
nothing beyond what the relational structure already captures.

**Caveats and next steps:**
- Label is log-probability-based, not true success/failure. The 0%
  top-1 accuracy is itself a finding: GPT-2 small does not solve bAbI
  zero-shot at the token prediction level.
- Results are from Task 1 only (single supporting fact). Tasks 2 and 3
  (multi-hop reasoning) are the natural next test — relational features
  may show stronger advantage when more supporting facts must be
  integrated.
- Sample size: 500 problems, 100 test. Results should be replicated on
  larger samples.
- Layer 11 (final layer) only. Probing earlier layers is a natural
  extension.

---
### 4.5 Multi-hop extension (Thursday 10, 2026-08-06)
> **⚠ SUPERSEDED — READ SECTION 4.7 FIRST.** These results share the
> question-access confound described in Section 4.7. The Task 3
> advantage reported in Section 4.6 does not survive the control.

**Configuration:** GPT-2 small, layer 11, bAbI Tasks 1 and 2, 300
problems per task, 80/20 split, fixed-dimension features for both
probe types (coordinate: mean-pooled activations, 768 dims;
relational: averaged pairwise relations, 770 dims) to avoid a
dimensionality confound across tasks.

**Results:**

| Task | Avg hops | Coord acc | Rel acc | Acc gap | AUC gap |
|---|---|---|---|---|---|
| bAbI 1 | 1.00 | 63.3% | 63.3% | +0.0pp | −0.006 |
| bAbI 2 | 2.00 | 80.0% | 76.7% | −3.3pp | −0.061 |

**Pre-registered prediction (Section 1.1):** SUPPORTED on both tasks.
Relational probe is within 5pp of coordinate probe in both cases.

**Secondary hypothesis (relational advantage grows with hop count):**
INCONCLUSIVE. The observed direction is opposite to the hypothesis,
but the test set is 60 examples per task. Standard error at this
sample size is approximately ±5.6pp, so the observed 3.3pp gap is
within one standard error of zero. This experiment lacks the
statistical power to distinguish a real effect from sampling noise.

**Unexplained observation:** both probes perform substantially better
on Task 2 (2 hops) than Task 1 (1 hop) — 80% vs 63% for coordinate,
76.7% vs 63.3% for relational. Harder reasoning task, more probeable
activations. No confident explanation at present. [uncertain]
Candidate causes: longer stories may produce greater spread in
answer log-probability, making the median split more separable; or
multi-hop problems may produce more distinctive activation
signatures. Worth investigating.

**Next step:** cache extracted activations to disk so probe
experiments can be run at 2,000+ problems per task without
recomputing activations each time. Current bottleneck is compute,
not method.

---

### 4.6 K-fold cross-validation and ablation (Thursday 11–12, 2026-08-06/13)

> **⚠ PARTIALLY SUPERSEDED — READ SECTION 4.7.** The Task 3 result
> below is explained by the question-access confound. The ablation
> finding stands.

**Method improvements.** Replaced the single 80/20 split with 5-fold
stratified cross-validation (all 1000 problems evaluated, not 200)
and added paired bootstrap on the accuracy difference. The pairing
matters: both probes are evaluated on identical examples, so shared
problem difficulty cancels. Marginal confidence intervals had
overlapped and suggested "inconclusive"; the paired test on the same
data returned p < 0.0001.

**Correction to Section 4.5.** Task 2's apparent falsification
(−6.0pp, crossing the pre-registered 5pp threshold) was a
single-split artifact. Under 5-fold CV the gap is −2.5pp.

**Ablation of relational components (Thursday 12).** Two predictions
were logged before running: Julien predicted the scalars (cosine,
L2) carried the signal; Claude predicted the difference vector
dominated.

| Variant | dims | 1 hop | 2 hops | 3 hops |
|---|---|---|---|---|
| coord | 768 | 67.3% | 75.3% | 68.8% |
| cos_only | 1 | 59.0% | 49.3% | 50.0% |
| l2_only | 1 | 59.1% | 53.9% | 53.9% |
| scalars (averaged) | 2 | 58.7% | 57.4% | 52.7% |
| diff_only | 768 | 67.4% | 70.8% | 74.3% |
| full_rel | 770 | 67.3% | 72.8% | 74.4% |

**Finding.** `diff_only` matches `full_rel` within 0.1pp on every
task. The averaged scalars contribute nothing detectable; cosine
similarity on Task 3 scored 50.0% with AUC 0.500, exactly chance.
Claude's prediction was correct.

**Why this weakened the thesis.** The difference vector
d = mean(vᵢ − vⱼ) and the coordinate feature c = mean(vᵢ) are both
linear combinations of the same activation vectors in the same basis.
Neither is coordinate-free. The genuinely rotation-invariant features
were the ones that failed. The comparison was therefore never
"relations versus coordinates" but one linear contrast versus another.

**Design flaw identified.** The relational feature vector was 770
dimensions, of which 768 were the difference vector and 2 were
invariant scalars — each an *average* over all pairs (6 pairs on
Task 3), collapsing 6 measurements into 1 number. The invariant
components were never given a fair test.

---

### 4.7 Controls and fair comparisons (Thursday 13, 2026-08-20)

Three controls were run on the existing activation cache, all with
5-fold CV and paired bootstrap.

#### 4.7.1 Question-access control — the headline result does not survive

The original coordinate probe pooled only the supporting-sentence
activations (v₁…v_k). The original relational probe used those *and*
the question activation (q). Unequal information. `coord_with_q`
corrects this by mean-pooling all k+1 vectors.

| Task | diff_only vs coord (original) | diff_only vs coord_with_q (fair) |
|---|---|---|
| 1 | +0.1pp, p=0.897, ns | −0.1pp, p=0.942, ns |
| 2 | −4.5pp, p=0.0016, **sig** | −4.7pp, p=0.0016, **sig** |
| 3 | **+5.5pp, p=0.0004, sig** | **−0.7pp, p=0.582, ns** |

**The Task 3 finding reported in Sections 4.5–4.6 was an artifact of
question-vector access.** Mean-pooling gains 6.2pp on Task 3 simply
from including q (68.8% → 75.0%). With equal information, `diff_only`
(74.3%) is marginally *behind* `coord_with_q` (75.0%).

Task 2 shows differencing is significantly *worse* than averaging.
Across all three tasks with equal information: no advantage, a
significant disadvantage, no advantage. **The thesis is falsified in
both its strong form (Section 4.6: invariants at chance) and its weak
form (differencing does not beat averaging).**

#### 4.7.2 Repairing the invariant features

The Gram matrix G_ij = ⟨vᵢ, vⱼ⟩ contains every quantity invariant
under rotation of the basis [recalled — standard invariant theory
result for the orthogonal group]. Cosine similarity and squared L2
distance both derive from it:

- ‖vᵢ‖² = G_ii
- cos θ_ij = G_ij / √(G_ii · G_jj)
- ‖vᵢ − vⱼ‖² = G_ii + G_jj − 2G_ij

Task 3 invariant accuracy as the representation improves:

| Representation | dims | accuracy |
|---|---|---|
| averaged scalars (§4.6) | 2 | 52.7% |
| unaveraged per-pair scalars | 12 | 56.9% |
| Gram matrix | 10 | **59.6%** |

The averaging in the original design was destroying roughly 7
percentage points of signal, and the Gram matrix outperforms
hand-picked cosine+L2 on all three tasks (59.9 vs 58.7; 62.3 vs
58.2; 59.6 vs 56.9).

**Quantified gap.** Even repaired, rotation-invariant features reach
~60% while full 768-dimensional features reach ~75%. Invariants
carry real signal — clearly above the 50% chance baseline — but
substantially less than the basis-dependent representation. **This is
the central quantitative result of the program.**

#### 4.7.3 Dimension-matched comparison

To test whether invariants lose on information content or merely on
parameter count, the difference vector was randomly projected to the
same dimensionality as the Gram matrix.

| Task | dims | gram | diff_proj | difference |
|---|---|---|---|---|
| 1 | 3 | 59.9% | 61.7% | −1.8pp, p=0.322, ns |
| 2 | 6 | 62.3% | 58.6% | +3.7pp, p=0.0176 |
| 3 | 10 | 59.6% | 59.8% | −0.2pp, p=0.957, ns |

The Task 2 result favours invariants at equal dimensionality, but
**does not survive multiple-comparison correction** (≈9 comparisons
this session; Bonferroni threshold ≈0.0056, or 0.0167 correcting for
three tasks only). One task in three, suggestive, not a finding.

#### 4.7.4 Unpredicted observation: question representation and hop count

| Task | coord | coord_with_q | gain from q |
|---|---|---|---|
| 1 | 67.3% | 67.5% | +0.2pp |
| 2 | 75.3% | 75.5% | +0.2pp |
| 3 | 68.8% | 75.0% | **+6.2pp** |

The question activation is near-worthless on Tasks 1–2 and decisive
on Task 3. Task 3 questions carry a constraint absent from the
supporting facts — *"Where was the football **before the
bathroom**?"* — so the supporting sentences alone are insufficient.
[uncertain] This emerged from a control designed to test something
else and has not been independently verified.

#### 4.7.5 Program status

The original thesis is falsified. Retained findings:

1. Rotation-invariant activation geometry carries real but limited
   signal (~60% vs ~75% for full-dimensional features).
2. Feature construction matters substantially: averaging invariants
   across pairs destroyed ~7pp of signal.
3. The Gram matrix is the principled invariant representation and
   outperforms hand-picked scalars on all tasks.
4. Question representation contributes only at three hops.
   [uncertain]

Required before any publication claim: layer sweep (does the
invariant gap hold at all depths?) and replication on a second model
(Pythia-160M).

*Section 4.7 completed: Thursday 13 (2026-08-20).*

*Section 4 design finalized: Thursday 5 (2026-06-11).*
---

## 5. References [PLACEHOLDER]

To be populated as Section 3 is drafted. Every citation will be verified
against the primary source before inclusion.

---
