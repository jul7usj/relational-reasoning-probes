# milestone 1 — lexical baseline
# purpose: can surface features alone predict the label?
# thursday 14
#
# THE CONCERN: coord_with_q accuracy is nearly identical at layer 0
# (raw embeddings) and layer 11 (fully processed). if the label is
# predictable from surface statistics, our entire study may be
# measuring sentence length and word frequency rather than reasoning.
#
# features use NO activations at all - only the bAbI text.

import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'cache')
N_FOLDS     = 5
RANDOM_SEED = 42
TASKS       = [('qa1_train', 'qa1_train.txt'),
               ('qa2_train', 'qa2_train.txt'),
               ('qa3_train', 'qa3_train.txt')]


def lexical_features(problem):
    """Surface features only. No model, no activations."""
    story_words = " ".join(problem['story']).lower().split()
    q_words = problem['question'].lower().split()
    supp_words = " ".join(
        problem['supporting_sentences']
    ).lower().split()

    q_set = set(q_words)
    supp_set = set(supp_words)
    story_set = set(story_words)

    return np.array([
        len(story_words),                    # story length
        len(q_words),                        # question length
        len(problem['story']),               # n sentences in story
        len(problem['supporting_sentences']), # n supporting
        len(supp_words),                     # supporting length
        len(q_set & supp_set),               # q-supporting overlap
        len(q_set & story_set),              # q-story overlap
        len(story_set),                      # story vocab size
        len(problem['answer']),              # answer char length
        # position of last supporting sentence in story
        max(problem['supporting_ids']) / max(len(problem['story']), 1),
        # spread of supporting sentence positions
        (max(problem['supporting_ids']) -
         min(problem['supporting_ids'])) / max(len(problem['story']), 1),
    ], dtype=float)


def cv_accuracy(X, y):
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
        preds[te] = clf.predict(sc.transform(X[te]))
        probs[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return accuracy_score(y, preds), roc_auc_score(y, probs)


def main():
    print("="*66)
    print("MILESTONE 1 — LEXICAL BASELINE")
    print("="*66)
    print("Can surface features predict the label without activations?")
    print("If yes at ~70%, the study measures surface statistics.")
    print("If near 55%, activations carry real information.\n")

    for task_base, task_file in TASKS:
        cache_p = os.path.join(
            CACHE_DIR, f"{task_base}_alllayers.npz"
        )
        if not os.path.exists(cache_p):
            print(f"  No cache for {task_base}. Skipping.")
            continue

        log_probs = np.load(cache_p)['log_probs']
        n = len(log_probs)
        y = (log_probs >= np.median(log_probs)).astype(int)

        problems = parse_babi_file(
            os.path.join(DATA_DIR, task_file)
        )[:n]
        X = np.array([lexical_features(p) for p in problems])

        acc, auc = cv_accuracy(X, y)

        print(f"{'='*66}")
        print(f"TASK: {task_base}   n={n}")
        print(f"{'='*66}")
        print(f"  lexical baseline: acc={acc:.1%}  auc={auc:.3f} "
              f"({X.shape[1]} surface features, 0 activations)")

        # individual feature diagnostics
        names = ['story_len', 'q_len', 'n_sents', 'n_supp',
                 'supp_len', 'q_supp_overlap', 'q_story_overlap',
                 'story_vocab', 'answer_len', 'supp_position',
                 'supp_spread']
        print(f"\n  single-feature accuracies:")
        singles = []
        for j, nm in enumerate(names):
            a, _ = cv_accuracy(X[:, [j]], y)
            singles.append((a, nm))
        for a, nm in sorted(singles, reverse=True)[:5]:
            print(f"    {nm:<18}{a:.1%}")
        print()

    print("="*66)
    print("COMPARE AGAINST (layer 11, from earlier runs):")
    print("  qa1 coord_with_q 67.5% | gram 59.9%")
    print("  qa2 coord_with_q 75.5% | gram 62.3%")
    print("  qa3 coord_with_q 75.0% | gram 59.6%")
    print("="*66)


if __name__ == "__main__":
    main()