# milestone 1 — multi-hop probe
# purpose: generalize the relational probe to N supporting sentences,
#           compare relational vs coordinate across bAbI tasks 1, 2, 3
# thursday 10
#
# key design choice: BOTH feature types are fixed-dimension regardless
# of number of supporting sentences, so results are comparable across
# tasks (no dimensionality confound).
#   - coordinate: mean-pool supporting activations -> 768 dims
#   - relational: average pairwise relations across all supporting
#                 sentences + question context -> 770 dims

import torch
import numpy as np
import os
import sys
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file
from transformer_lens import HookedTransformer

# ── configuration ────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
LAYER       = 11
N_PROBLEMS  = 300     # per task
TEST_SIZE   = 0.2
RANDOM_SEED = 42
TASKS       = ['qa1_train.txt', 'qa2_train.txt']  # tasks 1 and 2 today
# ─────────────────────────────────────────────────────────────────────

def get_answer_log_prob(model, problem):
    """Log probability GPT-2 assigns to the correct answer token."""
    story_text = " ".join(problem['story'])
    prompt = story_text + " " + problem['question']
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[0, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)
    answer_tokens = model.to_tokens(
        " " + problem['answer'], prepend_bos=False
    )
    answer_token_id = answer_tokens[0, 0].item()
    return log_probs[answer_token_id].item()


def extract_activation(model, sentence, layer=LAYER):
    """Residual stream activation at final token. Shape (768,)."""
    tokens = model.to_tokens(sentence)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache['resid_post', layer][0, -1, :].cpu().numpy()


def build_coordinate_features(supp_activations):
    """
    Mean-pool supporting sentence activations.
    Fixed shape (768,) regardless of number of supporting sentences.
    """
    return np.mean(supp_activations, axis=0)


def pairwise_relation(a, b):
    """Relation between two activation vectors: [cos_sim, l2_dist, diff]."""
    cos_sim = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    )
    l2_dist = np.linalg.norm(a - b)
    diff = a - b
    return np.concatenate([[cos_sim], [l2_dist], diff])


def build_relational_features(supp_activations, context_activation):
    """
    Average pairwise relations across all supporting sentences and
    the question context.
    Fixed shape (770,) regardless of number of supporting sentences.
    """
    # all vectors to relate: supporting sentences + question context
    vectors = list(supp_activations) + [context_activation]

    # all unique pairs
    pairs = list(combinations(range(len(vectors)), 2))

    relations = []
    for i, j in pairs:
        relations.append(pairwise_relation(vectors[i], vectors[j]))

    # average across all pairs -> fixed dimension
    return np.mean(relations, axis=0)


def run_task(model, task_file):
    """Run the full probe pipeline on one task. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"TASK: {task_file}")
    print('='*60)

    problems = parse_babi_file(
        os.path.join(DATA_DIR, task_file)
    )[:N_PROBLEMS]
    print(f"Processing {len(problems)} problems.")

    # step 1: labels
    print("  Computing log probabilities...")
    log_probs = np.array([
        get_answer_log_prob(model, p) for p in problems
    ])
    median_lp = np.median(log_probs)
    labels = (log_probs >= median_lp).astype(int)
    print(f"  Labels: {labels.sum()} high / "
          f"{len(labels)-labels.sum()} low")

    # step 2: features
    print("  Extracting activations and building features...")
    coord_features = []
    rel_features = []
    n_supporting_list = []

    for p in problems:
        supp_acts = [
            extract_activation(model, s)
            for s in p['supporting_sentences']
        ]
        ctx_act = extract_activation(model, p['question'])

        coord_features.append(build_coordinate_features(supp_acts))
        rel_features.append(
            build_relational_features(supp_acts, ctx_act)
        )
        n_supporting_list.append(len(supp_acts))

    coord_features = np.array(coord_features)
    rel_features = np.array(rel_features)
    avg_supporting = np.mean(n_supporting_list)

    print(f"  Coord shape: {coord_features.shape}, "
          f"Rel shape: {rel_features.shape}")
    print(f"  Avg supporting sentences per problem: "
          f"{avg_supporting:.2f}")

    # step 3: split
    Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        coord_features, labels, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=labels
    )
    Xr_tr, Xr_te, _, _ = train_test_split(
        rel_features, labels, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=labels
    )

    # step 4: scale + train
    cs = StandardScaler()
    Xc_tr_s = cs.fit_transform(Xc_tr)
    Xc_te_s = cs.transform(Xc_te)
    rs = StandardScaler()
    Xr_tr_s = rs.fit_transform(Xr_tr)
    Xr_te_s = rs.transform(Xr_te)

    coord_probe = LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED
    ).fit(Xc_tr_s, y_tr)
    rel_probe = LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED
    ).fit(Xr_tr_s, y_tr)

    coord_acc = accuracy_score(y_te, coord_probe.predict(Xc_te_s))
    rel_acc = accuracy_score(y_te, rel_probe.predict(Xr_te_s))
    coord_auc = roc_auc_score(
        y_te, coord_probe.predict_proba(Xc_te_s)[:, 1]
    )
    rel_auc = roc_auc_score(
        y_te, rel_probe.predict_proba(Xr_te_s)[:, 1]
    )

    print(f"  Coordinate: acc={coord_acc:.1%}, auc={coord_auc:.3f}")
    print(f"  Relational: acc={rel_acc:.1%}, auc={rel_auc:.3f}")

    return {
        'task': task_file,
        'avg_supporting': avg_supporting,
        'coord_acc': coord_acc,
        'rel_acc': rel_acc,
        'coord_auc': coord_auc,
        'rel_auc': rel_auc,
    }


def main():
    print("="*60)
    print("MILESTONE 1 — MULTI-HOP RELATIONAL PROBE")
    print("="*60)
    print(f"Config: layer {LAYER}, {N_PROBLEMS} problems/task, "
          f"tasks {TASKS}")

    print("\nLoading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    print("Model loaded.")

    results = []
    for task_file in TASKS:
        results.append(run_task(model, task_file))

    # summary table
    print("\n" + "="*60)
    print("SUMMARY — DOES RELATIONAL ADVANTAGE GROW WITH HOPS?")
    print("="*60)
    print(f"\n{'Task':<14}{'AvgHops':<10}{'CoordAcc':<11}"
          f"{'RelAcc':<10}{'AccGap':<10}{'AUCGap':<10}")
    print("-"*60)
    for r in results:
        acc_gap = r['rel_acc'] - r['coord_acc']
        auc_gap = r['rel_auc'] - r['coord_auc']
        print(f"{r['task']:<14}{r['avg_supporting']:<10.2f}"
              f"{r['coord_acc']:<11.1%}{r['rel_acc']:<10.1%}"
              f"{acc_gap:<+10.1%}{auc_gap:<+10.3f}")
    print("-"*60)
    print("\nInterpretation: if AccGap and AUCGap increase from")
    print("task 1 to task 2, relational features gain advantage as")
    print("reasoning requires integrating more supporting facts.")
    print("="*60)


if __name__ == "__main__":
    main()