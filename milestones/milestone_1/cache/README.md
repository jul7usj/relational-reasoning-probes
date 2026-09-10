# Activation cache

**The `.npz` files in this directory are gitignored.** They are derived
data, 30–77 MB each, regenerable from the code in `../code/`.

## Regenerating
python ../code/cache_all_layers.py

Extracts residual stream activations at all 12 layers for bAbI tasks
1–3, 1000 problems each. Approximately 52 minutes on CPU (GPT-2
small, no GPU required). Produces:

- `qa1_train_alllayers.npz` (~31 MB)
- `qa2_train_alllayers.npz` (~42 MB)
- `qa3_train_alllayers.npz` (~77 MB)

Resumable — checkpoints every 100 problems and continues from the
last checkpoint if interrupted.

## Format

Each `.npz` contains:

| key | shape | contents |
|---|---|---|
| `question_acts` | (12, N, 768) | question activation per layer |
| `supp_acts_flat` | (12, total, 768) | supporting activations, flattened |
| `supp_lengths` | (N,) | supporting sentences per problem |
| `log_probs` | (N,) | log prob of correct answer token |

`supp_acts_flat` is flattened because supporting-sentence count varies
by task (1, 2, or 3 for tasks 1–3 respectively). Use `supp_lengths` to
regroup — see `load_all_layers()` in `../code/probe_layer_sweep.py`.

## Why all 12 layers

`run_with_cache` computes every layer's residual stream regardless.
The earlier single-layer version discarded 11 of 12. Saving all layers
costs disk space, not compute — throughput was essentially identical
(1.5 vs 1.46 problems/sec on task 1).