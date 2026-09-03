# milestone 1 — controls and fair comparisons
# purpose: test whether the task 3 relational advantage survives
#           proper controls
# thursday 13
#
# THREE EXPERIMENTS:
#
# 1. QUESTION-ACCESS CONTROL
#    Original coordinate probe saw only supporting vectors (v1..vk).
#    Original relational probe saw supporting vectors AND question (q).
#    That is unequal information. coord_with_q fixes it by mean-pooling
#    all k+1 vectors. If the task 3 advantage vanishes, our headline
#    result was an artifact of information access, not relational
#    structure.
#
# 2. DIMENSION-MATCHED INVARIANT TEST
#    gram (rotation-invariant, k(k+1)/2 dims) vs diff_proj
#    (basis-dependent, randomly projected to the SAME dims).
#    Tests: at equal dimensionality, which carries more signal?
#
# 3. UNAVERAGED SCALARS
#    Original averaged 6 cosines into 1 number and 6 L2s into 1.
#    scalars_full keeps them separate. Middle data point between
#    2 dims and the full gram matrix.
#
# NOTE: variants with per-pair features (scalars_full, gram) have
# task-dependent dimensionality, since pair count varies with hop
# count. Comparisons are WITHIN task only, never across tasks.

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


def gram_features(vectors):
    """
    Upper triangle (incl. diagonal) of the Gram matrix G_ij = <vi, vj>.
    Contains ALL rotation-invariant information about the vector set.
    Dimension: k(k+1)/2 for k vectors.
    """
    k = len(vectors)
    V = np.array(vectors)
    G = V @ V.T
    return np.array([G[i, j] for i in range(k) for j in range(i, k)])


def scalars_full(vectors):
    """
    Per-pair cosine and L2, kept SEPARATE (not averaged).
    Dimension: 2 * C(k,2).
    """
    pairs = list(combinations(range(len(vectors)), 2))
    cos_vals, l2_vals = [], []
    for i, j in pairs:
        a, b = vectors[i], vectors[j]
        cos_vals.append(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )
        l2_vals.append(np.linalg.norm(a - b))
    return np.concatenate([cos_vals, l2_vals])


def diff_avg(vectors):
    """Averaged pairwise difference vector. 768 dims."""
    pairs = list(combinations(range(len(vectors)), 2))
    return np.mean(
        [vectors[i] - vectors[j] for i, j in pairs], axis=0
    )


def build_features(supp_acts, ctx_act, proj_matrix, gram_dim):
    """Build every variant for one problem."""
    with_q = list(supp_acts) + [ctx_act]

    d_avg = diff_avg(with_q)

    return {
        # baselines
        'coord':        np.mean(supp_acts, axis=0),          # 768
        'coord_with_q': np.mean(with_q, axis=0),             # 768
        'diff_only':    d_avg,                               # 768
        # invariant variants
        'gram':         gram_features(with_q),               # k(k+1)/2
        'scalars_full': scalars_full(with_q),                # 2*C(k,2)
        # dimension-matched control
        'diff_proj':    d_avg @ proj_matrix,                 # gram_dim
    }


def cv_evaluate(X, y):
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
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    ca = (pred_a == y).astype(float)
    cb = (pred_b == y).astype(float)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = ca[idx].mean() - cb[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


VARIANTS = ['coord', 'coord_with_q', 'diff_only',
            'gram', 'scalars_full', 'diff_proj']


def run_task(task_base):
    cached = load_cache(task_base)
    if cached is None:
        print(f"  No cache for {task_base}.")
        return None

    q_acts, supp_groups, log_probs, lengths = cached
    n = len(log_probs)
    y = (log_probs >= np.median(log_probs)).astype(int)

    # k = supporting + question; assumes constant within task
    k = int(lengths[0]) + 1
    gram_dim = k * (k + 1) // 2
    n_pairs = k * (k - 1) // 2

    # fixed random projection: 768 -> gram_dim
    rng = np.random.default_rng(RANDOM_SEED)
    proj = rng.normal(0, 1 / np.sqrt(gram_dim), size=(768, gram_dim))

    feat_lists = {v: [] for v in VARIANTS}
    for i in range(n):
        f = build_features(supp_groups[i], q_acts[i], proj, gram_dim)
        for v in VARIANTS:
            feat_lists[v].append(f[v])
    features = {v: np.array(feat_lists[v]) for v in VARIANTS}

    print(f"\n{'='*72}")
    print(f"TASK: {task_base}   n={n}  hops={int(lengths[0])}  "
          f"vectors(k)={k}  pairs={n_pairs}")
    print('='*72)
    print(f"{'variant':<15}{'dims':<8}{'acc':<9}{'sd':<8}{'auc':<8}")
    print('-'*72)

    res = {}
    for v in VARIANTS:
        acc, sd, auc, preds = cv_evaluate(features[v], y)
        res[v] = {'acc': acc, 'sd': sd, 'auc': auc, 'preds': preds,
                  'dims': features[v].shape[1]}
        print(f"{v:<15}{res[v]['dims']:<8}{acc:<9.1%}"
              f"{sd:<8.1%}{auc:<8.3f}")

    # ── experiment 1: question-access control ────────────────────────
    print(f"\n  [1] QUESTION-ACCESS CONTROL")
    m, lo, hi, p = paired_bootstrap(
        y, res['diff_only']['preds'], res['coord']['preds']
    )
    sig1 = not (lo <= 0 <= hi)
    print(f"      diff_only vs coord (original, unequal info):")
    print(f"        {m:+.1%}  CI[{lo:+.1%},{hi:+.1%}]  "
          f"p={p:.4f}  sig={'YES' if sig1 else 'no'}")

    m2, lo2, hi2, p2 = paired_bootstrap(
        y, res['diff_only']['preds'], res['coord_with_q']['preds']
    )
    sig2 = not (lo2 <= 0 <= hi2)
    print(f"      diff_only vs coord_with_q (FAIR, equal info):")
    print(f"        {m2:+.1%}  CI[{lo2:+.1%},{hi2:+.1%}]  "
          f"p={p2:.4f}  sig={'YES' if sig2 else 'no'}")
    if sig1 and not sig2:
        print(f"      -> ORIGINAL RESULT WAS AN ARTIFACT of question access")
    elif sig2:
        print(f"      -> result SURVIVES the fair control")
    else:
        print(f"      -> neither comparison significant")

    # ── experiment 2: dimension-matched invariant test ───────────────
    print(f"\n  [2] DIMENSION-MATCHED INVARIANT TEST ({gram_dim} dims each)")
    m3, lo3, hi3, p3 = paired_bootstrap(
        y, res['gram']['preds'], res['diff_proj']['preds']
    )
    sig3 = not (lo3 <= 0 <= hi3)
    print(f"      gram (invariant) vs diff_proj (basis-dependent):")
    print(f"        {m3:+.1%}  CI[{lo3:+.1%},{hi3:+.1%}]  "
          f"p={p3:.4f}  sig={'YES' if sig3 else 'no'}")
    if sig3 and m3 > 0:
        print(f"      -> invariant structure WINS at equal dims")
    elif sig3 and m3 < 0:
        print(f"      -> basis-dependent structure wins at equal dims")
    else:
        print(f"      -> indistinguishable at equal dims")

    # ── experiment 3: unaveraged scalars ─────────────────────────────
    print(f"\n  [3] UNAVERAGED SCALARS")
    print(f"      scalars_full ({res['scalars_full']['dims']} dims): "
          f"{res['scalars_full']['acc']:.1%}")
    print(f"      gram ({res['gram']['dims']} dims):         "
          f"{res['gram']['acc']:.1%}")
    print(f"      (thursday 12 averaged scalars, 2 dims, "
          f"scored 52.7% on task 3)")

    return {'task': task_base, 'hops': int(lengths[0]), 'k': k,
            'gram_dim': gram_dim, 'res': res,
            'q_control_survives': sig2,
            'invariant_wins': sig3 and m3 > 0}


def main():
    print("="*72)
    print("MILESTONE 1 — CONTROLS AND FAIR COMPARISONS")
    print("="*72)
    print("Experiments:")
    print("  1. question-access control (could invalidate headline result)")
    print("  2. dimension-matched invariant vs basis-dependent")
    print("  3. unaveraged scalars")
    print(f"\nLayer {LAYER}, {N_FOLDS}-fold CV, {N_BOOT} bootstrap, "
          f"seed {RANDOM_SEED}")

    results = [r for r in (run_task(t) for t in TASKS) if r]
    if not results:
        print("\nNo cache found.")
        return

    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"\n{'variant':<15}" + "".join(
        f"{r['task'][:3]+'('+str(r['hops'])+'h)':<13}" for r in results
    ))
    print('-'*72)
    for v in VARIANTS:
        row = f"{v:<15}"
        for r in results:
            row += (f"{r['res'][v]['acc']:.1%}"
                    f"/{r['res'][v]['dims']}d").ljust(13)
        print(row)
    print('-'*72)

    print("\nVERDICTS:")
    for r in results:
        print(f"  {r['task']} ({r['hops']}h):")
        print(f"    diff advantage survives question control: "
              f"{r['q_control_survives']}")
        print(f"    invariant beats basis-dependent at equal dims: "
              f"{r['invariant_wins']}")
    print("="*72)


if __name__ == "__main__":
    main()