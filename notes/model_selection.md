# Model and Benchmark Selection — Milestone 1

**Status:** decided Thursday 3 (2026-06-04). Locked for milestone 1.

---

## Model: GPT-2 small (124M parameters)

**Source:** OpenAI, 2019. Openly available via HuggingFace
(`openai-community/gpt2`).

**Architecture:** 12 layers, 12 attention heads, 768-dimensional hidden
state, ~124M parameters.

**Why this model:**

- The most-studied small transformer in mechanistic interpretability
  literature. A reviewer will immediately recognize it and trust the
  setup.
- TransformerLens has native support for GPT-2 small, meaning activation
  extraction and hook points are pre-built. We do not need to write
  custom activation extraction code from scratch.
- Runs comfortably on CPU for inference. Activation extraction on a
  12GB RAM laptop is feasible for small batches.
- Being old and imperfect at reasoning is not a disqualifier — for a
  probe, we need variance in reasoning step success, not a model that
  gets everything right.

**Hardware feasibility:** GPT-2 small model weights are approximately
500MB. Activation extraction for small batches (10–50 problems) requires
well under 12GB RAM. [assumption — to be confirmed when code starts]

**Backup model:** Pythia-160M (EleutherAI, 2023,
`EleutherAI/pythia-160m`). Also supported by TransformerLens. Switch to
this if GPT-2 small presents unexpected technical issues.

---

## Benchmark: bAbI tasks 1–3

**Source:** Weston et al., Facebook AI Research, 2015. Openly available
via HuggingFace datasets (`facebook/babi_qa`). [recalled — to be
verified when loading data]

**What bAbI is:** 20 toy reasoning tasks of increasing complexity. Tasks
1–3 cover:

- Task 1: single supporting fact (one-hop reasoning)
- Task 2: two supporting facts (two-hop reasoning)
- Task 3: three supporting facts (three-hop reasoning)

Each problem provides a set of context sentences, a question, and an
answer. The supporting facts that are necessary to answer the question
are explicitly labeled in the dataset.

**Why tasks 1–3 specifically:**

- Simple enough that "reasoning step" is unambiguous — each supporting
  fact is one reasoning step.
- GPT-2 small shows real variance on these tasks (it does not solve them
  perfectly), giving the probe enough signal to work with.
- Prior relational reasoning work (Santoro et al., 2017, Relation
  Networks) used bAbI, so our results sit in a known context.
- The full dataset is tiny (~1MB), loads in seconds, no compute overhead.

**What "reasoning step success" means for our probe:**

A reasoning step is the model's processing of one supporting fact
necessary to answer the question. Step success means the model's
activations at that step carry enough information to predict the correct
answer. The probe will be trained to predict step success from relational
features of the activations, compared against a baseline probe trained
on absolute activation values.

**Why not a custom dataset:**

A custom dataset would raise a reviewer question we don't want raised —
"was the dataset designed to make the probe look good?" bAbI sidesteps
this entirely. The control advantage of a custom dataset is not
meaningful enough at milestone 1 scale to outweigh the credibility cost.

**Why not GSM8K:**

GPT-2 small performs near-zero on math word problems. Not enough variance
in reasoning step success to probe against.

**Why not bAbI tasks 4–20:**

Tasks 4–20 introduce counting, lists, path-finding, and other structures
that complicate the definition of "reasoning step" without adding
scientific value at milestone 1 scale. We scope to tasks 1–3 and expand
only if results warrant it.

---

## Open questions (to resolve before probe code starts)

1. Confirm exact HuggingFace dataset name and loading syntax for bAbI.
2. Confirm TransformerLens installation and GPT-2 small hook points on
   this laptop's Python environment.
3. Confirm exact memory usage during activation extraction for batch
   size 10–50 on 12GB RAM.
4. Define operationally: what is a "relational feature" of activations
   in the context of bAbI tasks 1–3? (This is the core design question
   for the probe — to be drafted in milestone_1/README.md next Thursday.)

---

## Decision log

- Custom dataset vs. bAbI: chose bAbI for reviewer credibility and
  faster setup. Custom dataset deferred to a possible milestone 2
  variant if bAbI results are strong.
- GPT-2 small vs. Pythia-160M: chose GPT-2 small for density of prior
  mech-interp literature and TransformerLens native support.
- GPT-2 medium ruled out: compute constraint (345M parameters, too large
  for comfortable activation extraction on 12GB RAM laptop).

*Decided: Thursday 3 (2026-06-04)*