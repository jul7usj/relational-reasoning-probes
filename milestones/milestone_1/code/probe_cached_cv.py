# milestone 1 — cached probe with k-fold CV and paired bootstrap
# purpose: resolve the hop-scaling question with proper statistical power
# thursday 11
#
# two methodological fixes over probe_cached.py:
#   1. 5-fold cross-validation instead of a single 80/20 split
#      -> every problem used for both training and testing
#   2. paired bootstrap on the DIFFERENCE between probes
#      -> exploits that both probes see identical test examples,
#         which marginal CIs throw away
#
# also fixes a flaw in probe_cached.py's summary, which compared only
# task 1 vs task 3 endpoints and ignored the non-monotonic middle.

import numpy as np
import os
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ── configuration ────────────────────────────────────────────────────
CACHE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'cache')
LAYER       = 11
N_FOLDS     = 5
N_BOOT      = 5000
RANDOM_SEED = 42
TASKS       = ['qa1_train', 'qa2_train', 'qa3_train']
# ─────────────────────────────────────────────────────────────────────


def load_cache(task_base):
    path = os.path.join(CACHE_DIR, f"{task_base}_layer{LAYER}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    q_acts, flat = d['question_acts'], d['supp_acts_flat']
    lengths, log_probs = d['supp_lengths'], d['log_probs']
    supp_groups, idx = [], 0
    for n in lengths:
        supp_groups.append(flat[idx:idx + int(n)])
        idx += int(n)
    return q_acts, supp_groups, log_probs, lengths


def build_coordinate_features(supp_acts):
    return np.mean(supp_acts, axis=0)


def pairwise_relation(a, b):
    cos_sim = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    )
    return np.concatenate([[cos_sim], [np.linalg.norm(a - b)], a - b])


def build_relational_features(supp_acts, ctx_act):
    vectors = list(supp_acts) + [ctx_act]
    pairs = list(combinations(range(len(vectors)), 2))
    return np.mean(
        [pairwise_relation(vectors[i], vectors[j]) for i, j in pairs],
        axis=0
    )


def cv_predictions(X, y):
    """
    5-fold CV. Returns out-of-fold predictions and probabilities
    for every example, plus per-fold accuracies.
    """
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y))
    fold_accs = []

    for tr_idx, te_idx in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])

        clf = LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ).fit(X_tr, y[tr_idx])

        preds[te_idx] = clf.predict(X_te)
        probs[te_idx] = clf.predict_proba(X_te)[:, 1]
        fold_accs.append(accuracy_score(y[te_idx], preds[te_idx]))

    return preds, probs, np.array(fold_accs)


def paired_bootstrap_diff(y, pred_a, pred_b, n_boot=N_BOOT):
    """
    Bootstrap the DIFFERENCE in accuracy (a - b) on paired predictions.
    Returns (mean_diff, ci_low, ci_high, p_two_sided).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    correct_a = (pred_a == y).astype(float)
    correct_b = (pred_b == y).astype(float)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(correct_a[idx].mean() - correct_b[idx].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: fraction of resamples on the far side of zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


def run_task(task_base):
    cached = load_cache(task_base)
    if cached is None:
        print(f"  No cache for {task_base}. Skipping.")
        return None

    q_acts, supp_groups, log_probs, lengths = cached
    n = len(log_probs)
    labels = (log_probs >= np.median(log_probs)).astype(int)

    coord = np.array([
        build_coordinate_features(supp_groups[i]) for i in range(n)
    ])
    rel = np.array([
        build_relational_features(supp_groups[i], q_acts[i])
        for i in range(n)
    ])

    c_pred, c_prob, c_folds = cv_predictions(coord, labels)
    r_pred, r_prob, r_folds = cv_predictions(rel, labels)

    c_acc, r_acc = c_folds.mean(), r_folds.mean()
    c_auc = roc_auc_score(labels, c_prob)
    r_auc = roc_auc_score(labels, r_prob)

    # paired bootstrap on relational - coordinate
    mdiff, lo, hi, p = paired_bootstrap_diff(labels, r_pred, c_pred)
    significant = not (lo <= 0 <= hi)

    print(f"\n{'='*64}")
    print(f"TASK: {task_base}   n={n}, {N_FOLDS}-fold CV "
          f"(all {n} used for eval)")
    print('='*64)
    print(f"  Avg supporting sentences: {np.mean(lengths):.2f}")
    print(f"  Coordinate: acc={c_acc:.1%} "
          f"(fold sd {c_folds.std():.1%})  auc={c_auc:.3f}")
    print(f"  Relational: acc={r_acc:.1%} "
          f"(fold sd {r_folds.std():.1%})  auc={r_auc:.3f}")
    print(f"\n  Paired difference (relational - coordinate):")
    print(f"    point estimate: {mdiff:+.1%}")
    print(f"    95% CI:         [{lo:+.1%}, {hi:+.1%}]")
    print(f"    p (two-sided):  {p:.4f}")
    print(f"    significant:    {significant}")

    return {
        'task': task_base, 'n': n,
        'hops': np.mean(lengths),
        'c_acc': c_acc, 'r_acc': r_acc,
        'c_auc': c_auc, 'r_auc': r_auc,
        'diff': mdiff, 'lo': lo, 'hi': hi,
        'p': p, 'sig': significant,
    }


def main():
    print("="*64)
    print("MILESTONE 1 — CACHED PROBE, K-FOLD CV + PAIRED BOOTSTRAP")
    print("="*64)
    print(f"Layer {LAYER}, {N_FOLDS}-fold CV, "
          f"{N_BOOT} bootstrap resamples")

    results = [r for r in (run_task(t) for t in TASKS) if r]
    if not results:
        print("\nNo cache found. Run cache_activations.py first.")
        return

    print("\n" + "="*64)
    print("SUMMARY")
    print("="*64)
    print(f"\n{'Task':<12}{'Hops':<7}{'Coord':<9}{'Rel':<9}"
          f"{'Diff':<9}{'95% CI':<20}{'Sig':<6}")
    print("-"*72)
    for r in results:
        ci = f"[{r['lo']:+.1%}, {r['hi']:+.1%}]"
        print(f"{r['task']:<12}{r['hops']:<7.2f}{r['c_acc']:<9.1%}"
              f"{r['r_acc']:<9.1%}{r['diff']:<+9.1%}{ci:<20}"
              f"{str(r['sig']):<6}")
    print("-"*72)

    # pre-registered prediction: relational >= coordinate - 5pp
    print("\nPRE-REGISTERED PREDICTION (Section 1.1):")
    print("  relational accuracy >= coordinate accuracy - 5pp")
    for r in results:
        verdict = "SUPPORTED" if r['diff'] >= -0.05 else "FALSIFIED"
        robust = ("robust" if (r['hi'] < -0.05 or r['lo'] > -0.05)
                  else "NOT robust - CI spans threshold")
        print(f"  {r['task']}: diff={r['diff']:+.1%} -> "
              f"{verdict} ({robust})")

    # secondary hypothesis, stated honestly across ALL tasks
    print("\nSECONDARY HYPOTHESIS (advantage grows with hop count):")
    diffs = [r['diff'] for r in results]
    hops = [r['hops'] for r in results]
    print(f"  Diffs by hop count: " + ", ".join(
        f"{h:.0f}hop={d:+.1%}" for h, d in zip(hops, diffs)
    ))
    monotonic = all(
        diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1)
    )
    print(f"  Monotonically increasing across ALL tasks: {monotonic}")
    if not monotonic:
        print("  -> Pattern is NON-MONOTONIC. The hypothesis predicts")
        print("     a monotonic increase. It does not hold.")
    n_sig = sum(r['sig'] for r in results)
    print(f"  Individually significant differences: "
          f"{n_sig}/{len(results)}")
    if n_sig == 0:
        print("  -> No task shows a statistically distinguishable")
        print("     difference. Hypothesis remains untestable at n="
              f"{results[0]['n']}.")
    print("="*64)


if __name__ == "__main__":
    main()