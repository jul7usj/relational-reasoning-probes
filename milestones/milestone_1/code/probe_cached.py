# milestone 1 — cached probe
# purpose: run relational vs coordinate probes on cached activations
#           across bAbI tasks 1, 2, 3 — experiments in seconds
# thursday 11
#
# reads .npz caches written by cache_activations.py
# both feature types fixed-dimension (no dimensionality confound)

import numpy as np
import os
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ── configuration ────────────────────────────────────────────────────
CACHE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'cache')
LAYER       = 11
TEST_SIZE   = 0.2
RANDOM_SEED = 42
TASKS       = ['qa1_train', 'qa2_train', 'qa3_train']
# ─────────────────────────────────────────────────────────────────────


def load_cache(task_base):
    """Load cached activations. Returns per-problem structures."""
    path = os.path.join(CACHE_DIR, f"{task_base}_layer{LAYER}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)

    q_acts   = d['question_acts']
    flat     = d['supp_acts_flat']
    lengths  = d['supp_lengths']
    log_probs = d['log_probs']

    # regroup supporting activations per problem
    supp_groups, idx = [], 0
    for n in lengths:
        supp_groups.append(flat[idx:idx + int(n)])
        idx += int(n)

    return q_acts, supp_groups, log_probs, lengths


def build_coordinate_features(supp_acts):
    """Mean-pool supporting activations. Fixed shape (768,)."""
    return np.mean(supp_acts, axis=0)


def pairwise_relation(a, b):
    cos_sim = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    )
    l2_dist = np.linalg.norm(a - b)
    return np.concatenate([[cos_sim], [l2_dist], a - b])


def build_relational_features(supp_acts, ctx_act):
    """Average pairwise relations. Fixed shape (770,)."""
    vectors = list(supp_acts) + [ctx_act]
    pairs = list(combinations(range(len(vectors)), 2))
    relations = [
        pairwise_relation(vectors[i], vectors[j]) for i, j in pairs
    ]
    return np.mean(relations, axis=0)


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=RANDOM_SEED):
    """Bootstrap 95% CI on accuracy."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def run_task(task_base):
    cached = load_cache(task_base)
    if cached is None:
        print(f"  No cache found for {task_base}. Skipping.")
        return None

    q_acts, supp_groups, log_probs, lengths = cached
    n = len(log_probs)

    # labels: above-median log prob
    median_lp = np.median(log_probs)
    labels = (log_probs >= median_lp).astype(int)

    # build features
    coord = np.array([
        build_coordinate_features(supp_groups[i]) for i in range(n)
    ])
    rel = np.array([
        build_relational_features(supp_groups[i], q_acts[i])
        for i in range(n)
    ])

    # split
    Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        coord, labels, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=labels
    )
    Xr_tr, Xr_te, _, _ = train_test_split(
        rel, labels, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=labels
    )

    # scale + train
    cs, rs = StandardScaler(), StandardScaler()
    Xc_tr_s, Xc_te_s = cs.fit_transform(Xc_tr), cs.transform(Xc_te)
    Xr_tr_s, Xr_te_s = rs.fit_transform(Xr_tr), rs.transform(Xr_te)

    cp = LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED
    ).fit(Xc_tr_s, y_tr)
    rp = LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED
    ).fit(Xr_tr_s, y_tr)

    c_pred, r_pred = cp.predict(Xc_te_s), rp.predict(Xr_te_s)
    c_acc, r_acc = (
        accuracy_score(y_te, c_pred), accuracy_score(y_te, r_pred)
    )
    c_auc = roc_auc_score(y_te, cp.predict_proba(Xc_te_s)[:, 1])
    r_auc = roc_auc_score(y_te, rp.predict_proba(Xr_te_s)[:, 1])

    c_lo, c_hi = bootstrap_ci(y_te, c_pred)
    r_lo, r_hi = bootstrap_ci(y_te, r_pred)

    print(f"\n{'='*60}")
    print(f"TASK: {task_base}   (n={n}, test={len(y_te)})")
    print('='*60)
    print(f"  Avg supporting sentences: {np.mean(lengths):.2f}")
    print(f"  Labels: {labels.sum()} high / {n - labels.sum()} low")
    print(f"  Coordinate: acc={c_acc:.1%} "
          f"[{c_lo:.1%}, {c_hi:.1%}]  auc={c_auc:.3f}")
    print(f"  Relational: acc={r_acc:.1%} "
          f"[{r_lo:.1%}, {r_hi:.1%}]  auc={r_auc:.3f}")

    return {
        'task': task_base,
        'n': n,
        'n_test': len(y_te),
        'avg_hops': np.mean(lengths),
        'c_acc': c_acc, 'r_acc': r_acc,
        'c_auc': c_auc, 'r_auc': r_auc,
        'c_ci': (c_lo, c_hi), 'r_ci': (r_lo, r_hi),
    }


def main():
    print("="*60)
    print("MILESTONE 1 — CACHED PROBE (tasks 1-3)")
    print("="*60)
    print(f"Layer {LAYER}, test split {TEST_SIZE}, "
          f"bootstrap 95% CI on accuracy")

    results = [r for r in (run_task(t) for t in TASKS) if r]

    if not results:
        print("\nNo cached data found. Run cache_activations.py first.")
        return

    print("\n" + "="*60)
    print("SUMMARY — HOP SCALING")
    print("="*60)
    print(f"\n{'Task':<12}{'Hops':<7}{'N':<7}{'CoordAcc':<11}"
          f"{'RelAcc':<11}{'AccGap':<10}{'AUCGap':<10}")
    print("-"*68)
    for r in results:
        print(f"{r['task']:<12}{r['avg_hops']:<7.2f}{r['n']:<7}"
              f"{r['c_acc']:<11.1%}{r['r_acc']:<11.1%}"
              f"{r['r_acc']-r['c_acc']:<+10.1%}"
              f"{r['r_auc']-r['c_auc']:<+10.3f}")
    print("-"*68)

    print("\nPre-registered prediction (Section 1.1):")
    print("  relational accuracy >= coordinate accuracy - 5pp")
    for r in results:
        gap = r['r_acc'] - r['c_acc']
        verdict = "SUPPORTED" if gap >= -0.05 else "FALSIFIED"
        print(f"  {r['task']}: gap={gap:+.1%} -> {verdict}")

    print("\nSecondary hypothesis (relational advantage grows with hops):")
    gaps = [r['r_acc'] - r['c_acc'] for r in results]
    if len(gaps) >= 2:
        direction = "increases" if gaps[-1] > gaps[0] else "decreases"
        print(f"  Gap {direction} from task 1 to task "
              f"{len(gaps)}: {gaps[0]:+.1%} -> {gaps[-1]:+.1%}")
        print("  Check CI overlap before drawing conclusions.")
    print("="*60)


if __name__ == "__main__":
    main()