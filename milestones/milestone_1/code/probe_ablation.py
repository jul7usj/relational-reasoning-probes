# milestone 1 — feature ablation
# purpose: determine which component of the relational feature vector
#           carries the reasoning signal
# thursday 12
#
# pre-registered predictions (logged before running):
#   Julien:  the two scalars (cos + L2) carry most of the signal
#   Claude:  the 768-dim difference vector dominates
#
# variants tested per task:
#   coord        - mean-pooled activations (768)      [baseline]
#   cos_only     - cosine similarity only (1)
#   l2_only      - L2 distance only (1)
#   scalars      - cosine + L2 (2)
#   diff_only    - difference vector only (768)
#   full_rel     - cos + L2 + diff (770)              [current relational]

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


def relation_components(a, b):
    """Return (cos_sim, l2_dist, diff_vector) for one pair."""
    cos_sim = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    )
    l2_dist = np.linalg.norm(a - b)
    return cos_sim, l2_dist, a - b


def build_all_features(supp_acts, ctx_act):
    """
    Build every feature variant for one problem.
    All averaged across pairs -> fixed dimension per variant.
    """
    vectors = list(supp_acts) + [ctx_act]
    pairs = list(combinations(range(len(vectors)), 2))

    cos_list, l2_list, diff_list = [], [], []
    for i, j in pairs:
        c, l, d = relation_components(vectors[i], vectors[j])
        cos_list.append(c)
        l2_list.append(l)
        diff_list.append(d)

    cos_avg  = np.mean(cos_list)
    l2_avg   = np.mean(l2_list)
    diff_avg = np.mean(diff_list, axis=0)

    return {
        'coord':     np.mean(supp_acts, axis=0),
        'cos_only':  np.array([cos_avg]),
        'l2_only':   np.array([l2_avg]),
        'scalars':   np.array([cos_avg, l2_avg]),
        'diff_only': diff_avg,
        'full_rel':  np.concatenate([[cos_avg], [l2_avg], diff_avg]),
    }


def cv_evaluate(X, y):
    """5-fold CV. Returns (mean_acc, fold_sd, auc, oof_predictions)."""
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y))
    fold_accs = []

    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ).fit(X_tr, y[tr])
        preds[te] = clf.predict(X_te)
        probs[te] = clf.predict_proba(X_te)[:, 1]
        fold_accs.append(accuracy_score(y[te], preds[te]))

    fold_accs = np.array(fold_accs)
    return (fold_accs.mean(), fold_accs.std(),
            roc_auc_score(y, probs), preds)


def paired_bootstrap(y, pred_a, pred_b, n_boot=N_BOOT):
    """Bootstrap difference (a - b). Returns (mean, lo, hi, p)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    ca = (pred_a == y).astype(float)
    cb = (pred_b == y).astype(float)
    diffs = np.array([
        ca[i].mean() - cb[i].mean()
        for i in (rng.integers(0, n, n) for _ in range(n_boot))
    ])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


VARIANTS = ['coord', 'cos_only', 'l2_only',
            'scalars', 'diff_only', 'full_rel']
DIMS = {'coord': 768, 'cos_only': 1, 'l2_only': 1,
        'scalars': 2, 'diff_only': 768, 'full_rel': 770}


def run_task(task_base):
    cached = load_cache(task_base)
    if cached is None:
        print(f"  No cache for {task_base}. Skipping.")
        return None

    q_acts, supp_groups, log_probs, lengths = cached
    n = len(log_probs)
    y = (log_probs >= np.median(log_probs)).astype(int)

    # build every variant for every problem
    feat_lists = {v: [] for v in VARIANTS}
    for i in range(n):
        feats = build_all_features(supp_groups[i], q_acts[i])
        for v in VARIANTS:
            feat_lists[v].append(feats[v])
    features = {v: np.array(feat_lists[v]) for v in VARIANTS}

    print(f"\n{'='*70}")
    print(f"TASK: {task_base}   n={n}, hops={np.mean(lengths):.0f}")
    print('='*70)
    print(f"{'variant':<12}{'dims':<7}{'acc':<9}{'sd':<8}{'auc':<8}")
    print('-'*70)

    results = {}
    for v in VARIANTS:
        acc, sd, auc, preds = cv_evaluate(features[v], y)
        results[v] = {'acc': acc, 'sd': sd, 'auc': auc, 'preds': preds}
        print(f"{v:<12}{DIMS[v]:<7}{acc:<9.1%}{sd:<8.1%}{auc:<8.3f}")

    # key comparisons vs coordinate baseline
    print(f"\n  Paired comparisons vs coordinate baseline:")
    for v in ['scalars', 'diff_only', 'full_rel']:
        m, lo, hi, p = paired_bootstrap(
            y, results[v]['preds'], results['coord']['preds']
        )
        sig = "YES" if not (lo <= 0 <= hi) else "no"
        print(f"    {v:<11} {m:+.1%}  "
              f"CI[{lo:+.1%},{hi:+.1%}]  p={p:.4f}  sig={sig}")

    # scalars vs full relational - the decisive test
    m, lo, hi, p = paired_bootstrap(
        y, results['scalars']['preds'], results['full_rel']['preds']
    )
    sig = "YES" if not (lo <= 0 <= hi) else "no"
    print(f"\n  DECISIVE TEST — scalars (2 dims) vs full_rel (770 dims):")
    print(f"    {m:+.1%}  CI[{lo:+.1%},{hi:+.1%}]  "
          f"p={p:.4f}  sig={sig}")
    if not (lo <= 0 <= hi):
        print(f"    -> distinguishable: 768 extra dims DO matter")
    else:
        print(f"    -> indistinguishable: 2 scalars match 770 dims")

    return {'task': task_base, 'hops': np.mean(lengths),
            'results': results, 'y': y}


def main():
    print("="*70)
    print("MILESTONE 1 — FEATURE ABLATION")
    print("="*70)
    print("Pre-registered predictions:")
    print("  Julien: the 2 scalars (cos + L2) carry most of the signal")
    print("  Claude: the 768-dim difference vector dominates")
    print(f"\nLayer {LAYER}, {N_FOLDS}-fold CV, {N_BOOT} bootstrap")

    all_results = [r for r in (run_task(t) for t in TASKS) if r]
    if not all_results:
        print("\nNo cache. Run cache_activations.py first.")
        return

    print("\n" + "="*70)
    print("SUMMARY — ACCURACY BY VARIANT AND TASK")
    print("="*70)
    header = f"\n{'variant':<12}{'dims':<7}"
    for r in all_results:
        header += f"{r['task'][:3]+'('+str(int(r['hops']))+'h)':<12}"
    print(header)
    print('-'*70)
    for v in VARIANTS:
        row = f"{v:<12}{DIMS[v]:<7}"
        for r in all_results:
            row += f"{r['results'][v]['acc']:<12.1%}"
        print(row)
    print('-'*70)

    print("\nVERDICT ON PREDICTIONS:")
    for r in all_results:
        sc = r['results']['scalars']['acc']
        df = r['results']['diff_only']['acc']
        fr = r['results']['full_rel']['acc']
        winner = "scalars" if sc >= df else "diff_vector"
        print(f"  {r['task']} ({int(r['hops'])}h): "
              f"scalars={sc:.1%}, diff={df:.1%}, full={fr:.1%} "
              f"-> {winner} stronger")
    print("="*70)


if __name__ == "__main__":
    main()