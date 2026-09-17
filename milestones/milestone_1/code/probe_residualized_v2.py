# milestone 1 — residualization v2 (nonlinear)
# purpose: fully remove the lexical confound from the label
# thursday 15
#
# v1 PROBLEM: linear residualization left lexical at 56.6% on task 3
# (target 50%). linear regression removes only LINEAR dependence; the
# binary label can still be predictable through nonlinear structure.
#
# v2: residualize against three models of increasing flexibility.
#   linear   - baseline, what v1 did
#   poly2    - degree-2 expansion (squares + interactions), 11 -> 77
#   gbr      - gradient boosting, fully nonlinear
#
# DIAGNOSTIC: a too-flexible residualizer strips real signal along with
# the confound. we want lexical -> ~50% while coord_with_q HOLDS. if
# both collapse, we over-removed.

import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file
from lexical_baseline import lexical_features

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'cache')
LAYER       = 11
N_FOLDS     = 5
N_BOOT      = 5000
RANDOM_SEED = 42
TASKS       = [('qa1_train', 'qa1_train.txt'),
               ('qa2_train', 'qa2_train.txt'),
               ('qa3_train', 'qa3_train.txt')]


def load_layer(task_base, layer=LAYER):
    path = os.path.join(CACHE_DIR, f"{task_base}_alllayers.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    q = d['question_acts'][layer]
    flat = d['supp_acts_flat'][layer]
    lengths, lps = d['supp_lengths'], d['log_probs']
    groups, idx = [], 0
    for n in lengths:
        n = int(n)
        groups.append([flat[idx + j] for j in range(n)])
        idx += n
    return q, groups, lengths, lps


def gram_features(vectors):
    V = np.array(vectors)
    G = V @ V.T
    k = len(vectors)
    return np.array([G[i, j] for i in range(k) for j in range(i, k)])


def cv_eval(X, y):
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
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


def residualize(L, log_probs, method):
    """Return residual of log_probs after removing lexical structure."""
    sc = StandardScaler()
    Ls = sc.fit_transform(L)

    if method == 'linear':
        model = LinearRegression().fit(Ls, log_probs)
        pred = model.predict(Ls)
    elif method == 'poly2':
        pf = PolynomialFeatures(degree=2, include_bias=False)
        Lp = pf.fit_transform(Ls)
        model = LinearRegression().fit(Lp, log_probs)
        pred = model.predict(Lp)
    elif method == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            random_state=RANDOM_SEED
        ).fit(Ls, log_probs)
        pred = model.predict(Ls)
    else:
        raise ValueError(method)

    ss_res = np.sum((log_probs - pred) ** 2)
    ss_tot = np.sum((log_probs - log_probs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return log_probs - pred, r2


def main():
    print("=" * 74)
    print("MILESTONE 1 — RESIDUALIZATION v2 (NONLINEAR)")
    print("=" * 74)
    print("Goal: lexical -> ~50% while coord_with_q HOLDS.")
    print("If both collapse, the residualizer removed real signal.\n")

    best = {}

    for task_base, task_file in TASKS:
        data = load_layer(task_base)
        if data is None:
            print(f"  No cache for {task_base}."); continue
        q_acts, groups, lengths, log_probs = data
        n = len(log_probs)

        problems = parse_babi_file(
            os.path.join(DATA_DIR, task_file)
        )[:n]
        L = np.array([lexical_features(p) for p in problems])

        # activation features (computed once)
        coord_q, gram = [], []
        for i in range(n):
            vecs = groups[i] + [q_acts[i]]
            coord_q.append(np.mean(vecs, axis=0))
            gram.append(gram_features(vecs))
        coord_q, gram = np.array(coord_q), np.array(gram)

        print("=" * 74)
        print(f"TASK: {task_base}   n={n}")
        print("=" * 74)
        print(f"  {'method':<10}{'R2':<8}{'lexical':<11}"
              f"{'coord_q':<11}{'gram':<11}{'verdict':<18}")
        print("  " + "-" * 68)

        rows = []
        for method in ['linear', 'poly2', 'gbr']:
            resid, r2 = residualize(L, log_probs, method)
            y = (resid >= np.median(resid)).astype(int)

            lex_acc, p_lex = cv_eval(L, y)
            c_acc, p_c = cv_eval(coord_q, y)
            g_acc, p_g = cv_eval(gram, y)

            clean = abs(lex_acc - 0.5) <= 0.03
            holds = c_acc >= 0.65
            if clean and holds:
                verdict = "CLEAN"
            elif clean and not holds:
                verdict = "over-removed"
            else:
                verdict = "confound remains"

            print(f"  {method:<10}{r2:<8.3f}{lex_acc:<11.1%}"
                  f"{c_acc:<11.1%}{g_acc:<11.1%}{verdict:<18}")
            rows.append({'method': method, 'r2': r2, 'y': y,
                         'lex': lex_acc, 'coord': c_acc,
                         'gram': g_acc, 'p_lex': p_lex,
                         'p_c': p_c, 'p_g': p_g,
                         'clean': clean, 'holds': holds})

        # pick the least aggressive method that is clean
        chosen = None
        for r in rows:
            if r['clean'] and r['holds']:
                chosen = r
                break
        if chosen is None:
            chosen = min(rows, key=lambda r: abs(r['lex'] - 0.5))
            note = " (none fully clean; closest to 50%)"
        else:
            note = ""

        print(f"\n  CHOSEN: {chosen['method']}{note}")
        m, lo, hi, p = paired_ci(
            chosen['y'], chosen['p_c'], chosen['p_lex']
        )
        print(f"    coord_with_q vs lexical: {m:+.1%} "
              f"CI[{lo:+.1%},{hi:+.1%}] p={p:.4f}")
        m2, lo2, hi2, p2 = paired_ci(
            chosen['y'], chosen['p_c'], chosen['p_g']
        )
        print(f"    coord_with_q vs gram:    {m2:+.1%} "
              f"CI[{lo2:+.1%},{hi2:+.1%}] p={p2:.4f}")
        print()

        best[task_base] = chosen

    print("=" * 74)
    print("SUMMARY — CHOSEN RESIDUALIZER PER TASK")
    print("=" * 74)
    print(f"\n{'task':<14}{'method':<10}{'lexical':<11}"
          f"{'coord_q':<11}{'gram':<11}{'gap':<10}")
    print("-" * 70)
    for t, r in best.items():
        print(f"{t:<14}{r['method']:<10}{r['lex']:<11.1%}"
              f"{r['coord']:<11.1%}{r['gram']:<11.1%}"
              f"{r['coord']-r['gram']:<+10.1%}")
    print("-" * 70)
    print("\nlexical near 50% = label is orthogonal to surface features")
    print("coord_q well above 50% = activations carry real signal")
    print("gap = the invariant deficit, on a clean label")
    print("=" * 74)


if __name__ == "__main__":
    main()