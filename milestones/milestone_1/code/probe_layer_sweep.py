# milestone 1 — layer sweep
# purpose: does the invariant gap (~60% vs ~75%) hold across all
#           12 layers, or does it vary with depth?
# thursday 14
#
# compares at every layer:
#   coord_with_q  - mean-pool all vectors incl. question (768 dims)
#   gram          - full Gram matrix, rotation-invariant (k(k+1)/2)
#   diff_proj     - difference vector projected to gram_dim (matched)
#
# the headline number is the GAP: coord_with_q - gram, per layer.

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

CACHE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'cache')
N_LAYERS    = 12
N_FOLDS     = 5
N_BOOT      = 2000
RANDOM_SEED = 42
TASKS       = ['qa1_train', 'qa2_train', 'qa3_train']


def load_all_layers(task_base):
    path = os.path.join(CACHE_DIR, f"{task_base}_alllayers.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (d['question_acts'], d['supp_acts_flat'],
            d['supp_lengths'], d['log_probs'])


def vectors_at_layer(q_acts, flat, lengths, layer, i, start_idx):
    """All vectors for problem i at one layer: supporting + question."""
    n = int(lengths[i])
    supp = [flat[layer, start_idx + j, :] for j in range(n)]
    return supp + [q_acts[layer, i, :]]


def gram_features(vectors):
    V = np.array(vectors)
    G = V @ V.T
    k = len(vectors)
    return np.array([G[i, j] for i in range(k) for j in range(i, k)])


def diff_avg(vectors):
    from itertools import combinations
    pairs = list(combinations(range(len(vectors)), 2))
    return np.mean([vectors[i] - vectors[j] for i, j in pairs], axis=0)


def cv_accuracy(X, y):
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    preds = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ).fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    return accuracy_score(y, preds), preds


def paired_ci(y, pa, pb, n_boot=N_BOOT):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    ca, cb = (pa == y).astype(float), (pb == y).astype(float)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = ca[idx].mean() - cb[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi


def run_task(task_base):
    data = load_all_layers(task_base)
    if data is None:
        print(f"  No all-layer cache for {task_base}.")
        return None
    q_acts, flat, lengths, log_probs = data

    n = len(log_probs)
    y = (log_probs >= np.median(log_probs)).astype(int)
    k = int(lengths[0]) + 1
    gram_dim = k * (k + 1) // 2

    rng = np.random.default_rng(RANDOM_SEED)
    proj = rng.normal(0, 1 / np.sqrt(gram_dim), size=(768, gram_dim))

    # precompute start index per problem
    starts, acc_idx = [], 0
    for i in range(n):
        starts.append(acc_idx)
        acc_idx += int(lengths[i])

    print(f"\n{'='*70}")
    print(f"TASK: {task_base}   n={n}  hops={int(lengths[0])}  "
          f"gram_dim={gram_dim}")
    print('='*70)
    print(f"{'layer':<8}{'coord_q':<11}{'gram':<11}"
          f"{'diff_proj':<12}{'GAP':<10}{'CI':<18}")
    print('-'*70)

    rows = []
    for L in range(N_LAYERS):
        Xc, Xg, Xd = [], [], []
        for i in range(n):
            vecs = vectors_at_layer(q_acts, flat, lengths, L, i,
                                    starts[i])
            Xc.append(np.mean(vecs, axis=0))
            Xg.append(gram_features(vecs))
            Xd.append(diff_avg(vecs) @ proj)
        Xc, Xg, Xd = np.array(Xc), np.array(Xg), np.array(Xd)

        ac, pc = cv_accuracy(Xc, y)
        ag, pg = cv_accuracy(Xg, y)
        ad, _  = cv_accuracy(Xd, y)

        gap, lo, hi = paired_ci(y, pc, pg)
        ci = f"[{lo:+.1%},{hi:+.1%}]"
        print(f"{L:<8}{ac:<11.1%}{ag:<11.1%}{ad:<12.1%}"
              f"{gap:<+10.1%}{ci:<18}")
        rows.append({'layer': L, 'coord_q': ac, 'gram': ag,
                     'diff_proj': ad, 'gap': gap})

    gaps = [r['gap'] for r in rows]
    print('-'*70)
    print(f"  gap range: {min(gaps):+.1%} to {max(gaps):+.1%}  "
          f"(layer {int(np.argmin(gaps))} to "
          f"{int(np.argmax(gaps))})")
    return {'task': task_base, 'rows': rows}


def main():
    print("="*70)
    print("MILESTONE 1 — LAYER SWEEP")
    print("="*70)
    print("Question: does the invariant gap hold at all depths?")
    print(f"{N_LAYERS} layers, {N_FOLDS}-fold CV, {N_BOOT} bootstrap")

    results = [r for r in (run_task(t) for t in TASKS) if r]
    if not results:
        print("\nNo all-layer caches. Run cache_all_layers.py first.")
        return

    print("\n" + "="*70)
    print("GAP BY LAYER (coord_with_q - gram)")
    print("="*70)
    print(f"\n{'layer':<8}" + "".join(
        f"{r['task'][:3]:<12}" for r in results))
    print('-'*50)
    for L in range(N_LAYERS):
        row = f"{L:<8}"
        for r in results:
            row += f"{r['rows'][L]['gap']:<+12.1%}"
        print(row)
    print('-'*50)

    print("\nINTERPRETATION:")
    for r in results:
        gaps = [x['gap'] for x in r['rows']]
        trend = ("widens" if gaps[-1] > gaps[0] else "narrows")
        print(f"  {r['task']}: gap {trend} with depth "
              f"({gaps[0]:+.1%} at L0 -> {gaps[-1]:+.1%} at L11), "
              f"min {min(gaps):+.1%} at L{int(np.argmin(gaps))}")
    print("\n  Constant gap -> invariants insufficient throughout.")
    print("  Varying gap  -> localizes where representations")
    print("                  become basis-tied.")
    print("="*70)


if __name__ == "__main__":
    main()