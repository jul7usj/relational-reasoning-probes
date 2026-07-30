# milestone 1 — probe
# purpose: train relational and coordinate probes on GPT-2 small
#           activations from bAbI task 1, test pre-registered prediction
# thursday 9
#
# pre-registered prediction (from report Section 4.1.5):
#   relational probe accuracy >= coordinate probe accuracy - 5 percentage points
#   on 20% held-out test split

import torch
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file
from transformer_lens import HookedTransformer

# ── configuration ────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
LAYER       = 11       # final layer of GPT-2 small
N_PROBLEMS  = 500      # number of task 1 problems to process
TEST_SIZE   = 0.2      # 80/20 train/test split
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────

def get_answer_token_id(model, answer):
    """Get the token ID for a single-word answer."""
    tokens = model.to_tokens(" " + answer, prepend_bos=False)
    return tokens[0, 0].item()



def get_answer_log_prob(model, problem):
    """
    Returns the log probability GPT-2 assigns to the correct answer token
    given the full story + question prompt.
    
    Used instead of top-1 exact match because GPT-2 small solves bAbI
    zero-shot at 0% top-1 accuracy — the model never predicts the bare
    answer word as its top-1 next token. Log probability captures
    graded reasoning quality even when exact match fails.
    """
    story_text = " ".join(problem['story'])
    prompt = story_text + " " + problem['question']

    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        logits = model(tokens)

    # log probabilities at final token position
    last_logits = logits[0, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)

    # log prob of the correct answer token
    # prepend space because GPT-2 tokenizes " bathroom" not "bathroom"
    answer_tokens = model.to_tokens(
        " " + problem['answer'], prepend_bos=False
    )
    answer_token_id = answer_tokens[0, 0].item()

    return log_probs[answer_token_id].item()


def extract_activation(model, sentence, layer=LAYER):
    """
    Extract residual stream activation at final token of a sentence.
    Returns tensor of shape (768,).
    """
    tokens = model.to_tokens(sentence)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache['resid_post', layer][0, -1, :].cpu().numpy()


def build_coordinate_features(activations):
    """
    Coordinate features: raw activation values concatenated.
    For task 1, each problem has exactly 1 supporting sentence.
    Shape: (768,)
    """
    return activations[0]  # single supporting sentence for task 1


def build_relational_features(activations, context_activation):
    """
    Relational features: relations between the supporting sentence
    activation and the question context activation.
    
    Features:
    - cosine similarity (scalar)
    - L2 distance (scalar)  
    - difference vector (768,)
    
    Total shape: (770,)
    """
    supp = activations[0]       # supporting sentence activation
    ctx  = context_activation   # question context activation

    # cosine similarity
    cos_sim = np.dot(supp, ctx) / (
        np.linalg.norm(supp) * np.linalg.norm(ctx) + 1e-8
    )

    # L2 distance
    l2_dist = np.linalg.norm(supp - ctx)

    # difference vector
    diff = supp - ctx

    return np.concatenate([[cos_sim], [l2_dist], diff])


def main():
    print("=" * 60)
    print("MILESTONE 1 — RELATIONAL PROBE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model:       GPT-2 small (124M)")
    print(f"  Benchmark:   bAbI Task 1 (single supporting fact)")
    print(f"  Layer:       {LAYER} (final layer)")
    print(f"  N problems:  {N_PROBLEMS}")
    print(f"  Test split:  {TEST_SIZE}")
    print()

    # load model
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    print("Model loaded.\n")

    # load data
    print("Loading bAbI task 1...")
    task1 = parse_babi_file(os.path.join(DATA_DIR, 'qa1_train.txt'))
    problems = task1[:N_PROBLEMS]
    print(f"Processing {len(problems)} problems.\n")

   # ── step 1: compute labels from log probabilities ─────────────────
    print("Step 1: Computing answer log probabilities...")
    print("  (Using log-prob label: GPT-2 small gets 0% top-1 accuracy")
    print("   on bAbI zero-shot — see results for details.)")
    
    log_probs = []
    for i, problem in enumerate(problems):
        if i % 50 == 0:
            print(f"  [{i}/{N_PROBLEMS}] computing log probs...")
        lp = get_answer_log_prob(model, problem)
        log_probs.append(lp)

    log_probs = np.array(log_probs)
    
    # split at median: above median = 1 (high), below = 0 (low)
    median_lp = np.median(log_probs)
    labels = (log_probs >= median_lp).astype(int)
    
    n_high = labels.sum()
    n_low  = len(labels) - n_high
    print(f"  Done.")
    print(f"  Log prob range: [{log_probs.min():.2f}, "
          f"{log_probs.max():.2f}]")
    print(f"  Median log prob: {median_lp:.2f}")
    print(f"  High (above median): {n_high}, Low (below median): {n_low}")
    print(f"  Note: GPT-2 small top-1 exact match accuracy = 0.0%")
    print(f"        (model never predicts bare answer word as top-1 token)\n")

    if n_high == 0 or n_low == 0:
        print("ERROR: all labels are the same class even after median split.")
        print("Something is wrong with the log prob computation.")
        return

    # ── step 2: extract activations ──────────────────────────────────
    print("Step 2: Extracting activations...")
    coord_features    = []
    relational_features = []

    for i, problem in enumerate(problems):
        if i % 50 == 0:
            print(f"  [{i}/{N_PROBLEMS}] extracting activations...")

        # supporting sentence activation
        supp_activations = [
            extract_activation(model, s)
            for s in problem['supporting_sentences']
        ]

        # question context activation
        question_activation = extract_activation(
            model, problem['question']
        )

        # build features
        coord_feat = build_coordinate_features(supp_activations)
        rel_feat   = build_relational_features(
            supp_activations, question_activation
        )

        coord_features.append(coord_feat)
        relational_features.append(rel_feat)

    coord_features      = np.array(coord_features)
    relational_features = np.array(relational_features)

    print(f"  Done.")
    print(f"  Coordinate feature shape:  {coord_features.shape}")
    print(f"  Relational feature shape:  {relational_features.shape}\n")

    # ── step 3: train/test split ──────────────────────────────────────
    print("Step 3: Train/test split...")
    X_coord_train, X_coord_test, y_train, y_test = train_test_split(
        coord_features, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    X_rel_train, X_rel_test, _, _ = train_test_split(
        relational_features, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    print(f"  Train: {len(y_train)} problems")
    print(f"  Test:  {len(y_test)} problems\n")

    # ── step 4: train probes ─────────────────────────────────────────
    # ── step 4: train probes ─────────────────────────────────────────
    print("Step 4: Scaling features and training probes...")

    # scale features — required for logistic regression convergence
    # especially important for coordinate features (raw 768-dim vectors
    # with large value variance)
    coord_scaler = StandardScaler()
    X_coord_train_scaled = coord_scaler.fit_transform(X_coord_train)
    X_coord_test_scaled  = coord_scaler.transform(X_coord_test)

    rel_scaler = StandardScaler()
    X_rel_train_scaled = rel_scaler.fit_transform(X_rel_train)
    X_rel_test_scaled  = rel_scaler.transform(X_rel_test)

    # coordinate probe
    coord_probe = LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED
    )
    coord_probe.fit(X_coord_train_scaled, y_train)
    coord_acc = accuracy_score(
        y_test, coord_probe.predict(X_coord_test_scaled)
    )
    try:
        coord_auc = roc_auc_score(
            y_test,
            coord_probe.predict_proba(X_coord_test_scaled)[:, 1]
        )
    except Exception:
        coord_auc = float('nan')

    # relational probe
    rel_probe = LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED
    )
    rel_probe.fit(X_rel_train_scaled, y_train)
    rel_acc = accuracy_score(
        y_test, rel_probe.predict(X_rel_test_scaled)
    )
    try:
        rel_auc = roc_auc_score(
            y_test,
            rel_probe.predict_proba(X_rel_test_scaled)[:, 1]
        )
    except Exception:
        rel_auc = float('nan')

    print(f"  Done.\n")

    # ── step 5: results ───────────────────────────────────────────────
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Coordinate probe accuracy:  {coord_acc:.1%}")
    print(f"  Relational probe accuracy:  {rel_acc:.1%}")
    print(f"\n  Coordinate probe AUC:       {coord_auc:.3f}")
    print(f"  Relational probe AUC:       {rel_auc:.3f}")

    gap = coord_acc - rel_acc
    print(f"\n  Gap (coord - relational):   {gap:.1%}")

    # ── step 6: pre-registered prediction verdict ─────────────────────
    print()
    print("=" * 60)
    print("PRE-REGISTERED PREDICTION VERDICT")
    print("=" * 60)
    print()
    print("  Prediction: relational probe accuracy >= ")
    print("              coordinate probe accuracy - 5pp")
    print()
    print("  Label definition (revised from Section 4.1.4):")
    print("  Above-median log probability of correct answer token.")
    print("  Revision reason: GPT-2 small top-1 accuracy = 0% on")
    print("  bAbI zero-shot. Log-prob captures graded reasoning quality.")
    print()

    threshold = coord_acc - 0.05
    if rel_acc >= threshold:
        verdict = "SUPPORTED"
        explanation = (
            f"Relational probe ({rel_acc:.1%}) is within 5pp of "
            f"coordinate probe ({coord_acc:.1%}). "
            f"Gap = {gap:.1%}."
        )
    else:
        verdict = "FALSIFIED"
        explanation = (
            f"Relational probe ({rel_acc:.1%}) falls more than 5pp "
            f"below coordinate probe ({coord_acc:.1%}). "
            f"Gap = {gap:.1%}."
        )

    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()