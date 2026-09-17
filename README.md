# Relational Reasoning Probes

A mechanistic interpretability study asking whether the
*rotation-invariant* geometry of transformer activations carries
information about reasoning, or whether the signal lives in the
basis-dependent representation.

**Status:** milestone 1 complete. Result below. Not published; this
repository is the artifact.

---

## The original thesis — falsified

The program began with a pre-registered claim (timestamped in this
repo before any experiment was run):

> The relational structure of activations across reasoning steps —
> not their coordinate values — carries the reasoning load in
> transformer language models.

**This was falsified by our own experiments.** Full record in
`report/v0_outline.md`, Sections 4.4–4.9.

---

## What was actually found

Probing GPT-2 small (124M) on bAbI tasks 1–3, predicting a
median-split label derived from the model's log-probability of the
correct answer token:

| Probe | Features | Task 1 | Task 2 | Task 3 |
|---|---|---|---|---|
| lexical baseline | 11 surface features, no activations | 47.5% | 49.0% | 47.2% |
| `coord_with_q` | mean-pooled activations, 768 dims | **73.8%** | **72.6%** | **71.8%** |
| `gram` | full Gram matrix, rotation-invariant | 58.4% | 60.9% | 55.7% |

**The invariant deficit is +11.7 to +16.1 percentage points at the
final layer, and significant at all 36 layer-task combinations
(12 layers × 3 tasks), range +8.0 to +22.5pp.** All p < 0.0001,
5-fold cross-validation, paired bootstrap.

The Gram matrix contains *every* quantity invariant under rotation of
the activation basis — cosine similarities and pairwise distances both
derive from it. So this is not a statement about one hand-picked
feature set. **Rotation-invariant geometry is measurably insufficient
to recover what the full representation encodes, at every depth of
the network.**

---

## How the thesis died

Three experiments, each documented with code and output:

**1. Ablation** (`probe_ablation.py`). The relational feature vector
was 770 dimensions: 768 for the difference vector, 2 for invariant
scalars. The difference vector alone matched the full set within
0.1pp on every task; cosine similarity scored exactly chance (50.0%,
AUC 0.500) on Task 3. But a difference vector `a − b` is still a
linear function of raw coordinates — so the comparison was never
"relational versus coordinate," it was one linear contrast versus
another.

**2. Question-access control** (`probe_controls.py`). The relational
probe saw the question activation; the coordinate probe did not. With
equal information the Task 3 advantage vanished (+5.5pp → −0.7pp,
p = 0.58). **Our headline result at the time was an artifact of
unequal information access.**

**3. Lexical baseline** (`lexical_baseline.py`). Eleven surface
features with zero activations beat the 768-dim probe on Task 1
(71.1% vs 67.5%). Sentence count alone scored 67.5%. Surface features
explained up to 73% of log-probability variance — longer stories
depress the probability of any given token, so the median split was
partly sorting by length.

---

## The recovery

Residualizing the label against lexical features
(`probe_residualized_v2.py`) removed the confound and **increased**
activation probe accuracy (67.5→73.8, 75.5→72.6, 75.0→71.8 after
choosing the appropriate residualizer per task). Lexical noise had
been obscuring real signal.

Validation: a lexical probe on the residualized label must score ~50%.
It scores 47.5%, 49.0%, 47.2%. Residualizer chosen per task as the
least aggressive method achieving this — linear regression for
Task 1, gradient boosting for Tasks 2–3.

---

## Limitations — stated plainly

- **GPT-2 small scores 0% on bAbI zero-shot.** The label is a
  confidence proxy, not a correctness measure. What it means to probe
  "reasoning quality" in a model that cannot do the task is an open
  question, and the honest answer is that this study does not measure
  reasoning directly.
- **`coord_with_q` is nearly flat across depth.** Layer 0 (raw
  embeddings, before any transformer computation) performs about as
  well as layer 11. The transformer's processing adds little for this
  label.
- **The `gram` probe is underpowered** — 3, 6, and 10 dimensions for
  tasks 1–3. Layer-to-layer variation in that column should not be
  over-interpreted.
- **One model, one benchmark, one label definition.** No replication.
- The model (2019) and benchmark (2015) are both dated.

---

## Repository structure
report/v0_outline.md full technical report, all sections
milestones/milestone_1/
code/ all experiment scripts
data/ bAbI tasks 1-3
cache/ activation caches (gitignored, see README)
notes/ working notes, decisions, future directions


Scripts run in order: `cache_all_layers.py` (extraction, ~52 min CPU),
then any probe script (seconds, reads the cache).

---

## Methodology notes

- Predictions pre-registered publicly before experiments were run.
- 5-fold stratified cross-validation throughout; a single 80/20 split
  produced a spurious falsification on Task 2 (−6.0pp) that
  cross-validation corrected to −2.5pp.
- Paired bootstrap on the accuracy *difference*, not marginal
  confidence intervals — both probes see identical test examples, and
  discarding that pairing is too conservative to detect real effects.
- Negative results and self-invalidating controls committed to the
  history rather than removed.

---

## Contact

Julien Rached Abboud — julienr.abboud@gmail.com
