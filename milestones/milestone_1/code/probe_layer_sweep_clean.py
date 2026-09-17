# milestone 1 — layer sweep on residualized labels
# purpose: rerun the 12-layer sweep with the lexical confound removed
# thursday 15
#
# last week's sweep (probe_layer_sweep.py) used the CONFOUNDED label.
# on task 1 that label was 36% explainable by story length alone.
# this rerun uses the residualized labels validated in
# probe_residualized_v2.py:
#   task 1 - linear residualization  (lexical 47.5%)
#   task 2 - gradient boosting        (lexical 49.0%)
#   task 3 - gradient boosting        (lexical 47.2%)

import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file
from lexical_baseline import lexical_features

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'cache')
N_LAYERS    = 12
N_FOLDS     = 5
N_BOOT      = 2000
RANDOM_SEED = 42

# residualizer chosen per task by probe_residualized_v2.py
TASKS = [('qa1_train', 'qa1_train.txt', 'linear'),
         ('qa2_train', 'qa2_train.txt', 'gbr'),
         ('qa3_train', 'qa3_train.txt', 'gbr')]


def load_all(task_base):
    path = os.path.join(CACHE_DIR, f"{task_base}_alllayers.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (d['question_acts'], d['supp_acts_flat'],
            d['supp_lengths'], d['log_probs'])


def residualize(L, log_probs, method):
    sc = StandardScaler()
    Ls = sc.fit_transform(L)
    if method == 'linear':
        pred = LinearRegression().fit(Ls, log_probs).predict(Ls)
    else:
        pred = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=RANDOM_SEED
        ).fit(Ls, log_probs).predict(Ls)
    return log_probs - pred


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
    return diffs.mean(), lo, hi


def run_task(task_base, task_file, method):
    data = load_all(task_base)
    if data is None:
        print(f"  No cache for {task_base}."); return None
    q_acts, flat, lengths, log_probs = data
    n = len(log_probs)

    problems = parse_babi_file(
        os.path.join(DATA_DIR, task_file)
    )[:n]
    L = np.array([lexical_features(p) for p in problems])

    resid = residualize(L, log_probs, method)
    y = (resid >= np.median(resid)).astype(int)

    lex_acc, _ = cv_eval(L, y)

    starts, acc = [], 0
    for i in range(n):
        starts.append(acc)
        acc += int(lengths[i])

    k = int(lengths[0]) + 1
    print(f"\n{'='*72}")
    print(f"TASK: {task_base}   n={n}  hops={int(lengths[0])}  "
          f"residualizer={method}  lexical={lex_acc:.1%}")
    print('='*72)
    print(f"{'layer':<8}{'coord_q':<11}{'gram':<11}"
          f"{'GAP':<10}{'CI':<20}{'sig':<5}")
    print('-'*72)

    rows = []
    for Lay in range(N_LAYERS):
        Xc, Xg = [], []
        for i in range(n):
            supp = [flat[Lay, starts[i] + j, :]
                    for j in range(int(lengths[i]))]
            vecs = supp + [q_acts[Lay, i, :]]
            Xc.append(np.mean(vecs, axis=0))
            Xg.append(gram_features(vecs))
        Xc, Xg = np.array(Xc), np.array(Xg)

        ac, pc = cv_eval(Xc, y)
        ag, pg = cv_eval(Xg, y)
        gap, lo, hi = paired_ci(y, pc, pg)
        sig = "YES" if lo > 0 else "no"
        print(f"{Lay:<8}{ac:<11.1%}{ag:<11.1%}{gap:<+10.1%}"
              f"{f'[{lo:+.1%},{hi:+.1%}]':<20}{sig:<5}")
        rows.append({'layer': Lay, 'coord': ac, 'gram': ag,
                     'gap': gap, 'sig': lo > 0})

    gaps = [r['gap'] for r in rows]
    n_sig = sum(r['sig'] for r in rows)
    print('-'*72)
    print(f"  gap range {min(gaps):+.1%} to {max(gaps):+.1%}   "
          f"significant at {n_sig}/{N_LAYERS} layers")
    return {'task': task_base, 'lex': lex_acc, 'rows': rows,
            'n_sig': n_sig}


def main():
    print("="*72)
    print("MILESTONE 1 — LAYER SWEEP ON CLEAN (RESIDUALIZED) LABELS")
    print("="*72)
    print("Supersedes probe_layer_sweep.py, which used the")
    print("lexically confounded label.")

    results = [r for r in
               (run_task(t, f, m) for t, f, m in TASKS) if r]
    if not results:
        print("\nNo caches found."); return

    print("\n" + "="*72)
    print("GAP BY LAYER (coord_with_q - gram), CLEAN LABELS")
    print("="*72)
    print(f"\n{'layer':<8}" + "".join(
        f"{r['task'][:3]:<12}" for r in results))
    print('-'*46)
    for Lay in range(N_LAYERS):
        row = f"{Lay:<8}"
        for r in results:
            row += f"{r['rows'][Lay]['gap']:<+12.1%}"
        print(row)
    print('-'*46)
    total_sig = sum(r['n_sig'] for r in results)
    print(f"\nSignificant at {total_sig}/{N_LAYERS*len(results)} "
          f"layer-task combinations.")
    for r in results:
        print(f"  {r['task']}: lexical baseline {r['lex']:.1%} "
              f"(clean), significant at {r['n_sig']}/{N_LAYERS}")
    print("="*72)


if __name__ == "__main__":
    main()