# milestone 1 — residualized label
# purpose: remove lexical confound from the label, rerun the
#           coordinate vs invariant comparison
# thursday 14
#
# THE PROBLEM (thursday 14, lexical_baseline.py):
#   our label = above/below median log-prob of the answer token.
#   longer stories -> more tokens -> lower prob for any given token.
#   so the median split partly sorts by length. on task 1, 11 surface
#   features scored 71.1% vs 67.5% for the 768-dim activation probe.
#
# THE FIX:
#   1. regress log_prob on the lexical features (linear regression)
#   2. keep the RESIDUAL = log_prob - lexical_prediction
#   3. split the residual at its median
#   the new label is orthogonal to lexical features by construction.
#
# then a lexical probe on the new label MUST score ~50%. that is the
# validity check. if it doesn't, the residualization failed.

import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

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
    q = d['question_acts'][layer]          # (N, 768)
    flat = d['supp_acts_flat'][layer]      # (total, 768)
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
    probs = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ).fit(sc.fit_transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        preds[te] = clf.predict(Xte)
        probs[te] = clf.predict_proba(Xte)[:, 1]
    return accuracy_score(y, preds), roc_auc_score(y, probs), preds


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


def main():
    print("="*70)
    print("MILESTONE 1 — RESIDUALIZED LABEL")
    print("="*70)
    print("Label = median split of (log_prob - lexical prediction).")
    print("Orthogonal to surface features by construction.\n")

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

        # ── residualize ──────────────────────────────────────────────
        sc = StandardScaler()
        Ls = sc.fit_transform(L)
        lin = LinearRegression().fit(Ls, log_probs)
        pred_lp = lin.predict(Ls)
        resid = log_probs - pred_lp
        r2 = lin.score(Ls, log_probs)

        y_old = (log_probs >= np.median(log_probs)).astype(int)
        y_new = (resid >= np.median(resid)).astype(int)
        agree = (y_old == y_new).mean()

        # ── features ─────────────────────────────────────────────────
        coord_q, gram = [], []
        for i in range(n):
            vecs = groups[i] + [q_acts[i]]
            coord_q.append(np.mean(vecs, axis=0))
            gram.append(gram_features(vecs))
        coord_q, gram = np.array(coord_q), np.array(gram)

        # ── evaluate ─────────────────────────────────────────────────
        lex_old, _, _ = cv_eval(L, y_old)
        lex_new, _, p_lex = cv_eval(L, y_new)
        c_old, _, _ = cv_eval(coord_q, y_old)
        c_new, _, p_c = cv_eval(coord_q, y_new)
        g_old, _, _ = cv_eval(gram, y_old)
        g_new, _, p_g = cv_eval(gram, y_new)

        print("="*70)
        print(f"TASK: {task_base}   n={n}")
        print("="*70)
        print(f"  lexical R^2 on log_prob: {r2:.3f}  "
              f"(variance explained by surface features)")
        print(f"  label agreement old vs new: {agree:.1%}\n")
        print(f"  {'probe':<16}{'old label':<13}{'new label':<13}"
              f"{'change':<10}")
        print("  " + "-"*52)
        print(f"  {'lexical':<16}{lex_old:<13.1%}{lex_new:<13.1%}"
              f"{lex_new-lex_old:<+10.1%}")
        print(f"  {'coord_with_q':<16}{c_old:<13.1%}{c_new:<13.1%}"
              f"{c_new-c_old:<+10.1%}")
        print(f"  {'gram':<16}{g_old:<13.1%}{g_new:<13.1%}"
              f"{g_new-g_old:<+10.1%}")

        # validity check
        print(f"\n  VALIDITY: lexical on new label = {lex_new:.1%} "
              f"(must be ~50%)")
        if abs(lex_new - 0.5) > 0.05:
            print(f"    WARNING: residualization incomplete.")
        else:
            print(f"    OK - confound removed.")

        # does anything survive?
        m, lo, hi, p = paired_ci(y_new, p_c, p_lex)
        print(f"\n  coord_with_q vs lexical (new label): "
              f"{m:+.1%} CI[{lo:+.1%},{hi:+.1%}] p={p:.4f}")
        print(f"    -> activations beat surface features: "
              f"{'YES' if lo > 0 else 'NO'}")

        m2, lo2, hi2, p2 = paired_ci(y_new, p_c, p_g)
        print(f"  coord_with_q vs gram (new label):    "
              f"{m2:+.1%} CI[{lo2:+.1%},{hi2:+.1%}] p={p2:.4f}")
        print(f"    -> invariant gap survives: "
              f"{'YES' if lo2 > 0 else 'NO'}")
        print()

    print("="*70)
    print("If coord_with_q drops to ~50% on the new label, the entire")
    print("activation signal was surface statistics.")
    print("If it stays well above 50%, we have a real result.")
    print("="*70)


if __name__ == "__main__":
    main()  