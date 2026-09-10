# milestone 1 — all-layer activation cache builder
# purpose: extract activations at EVERY layer in one pass
# thursday 14
#
# key insight: run_with_cache already computes all 12 layers.
# the original cache_activations.py threw away 11 of them.
# saving all layers costs disk space, not compute.
#
# saves per task, as .npz:
#   question_acts   (n_layers, N, 768)
#   supp_acts_flat  (n_layers, total_supporting, 768)
#   supp_lengths    (N,)
#   log_probs       (N,)
#
# resumable: re-running continues from the last checkpoint.

import torch
import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file
from transformer_lens import HookedTransformer

# ── configuration ────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'cache')
N_LAYERS   = 12                     # GPT-2 small: layers 0-11
N_PROBLEMS = 1000
TASKS      = ['qa1_train.txt', 'qa2_train.txt', 'qa3_train.txt']
SAVE_EVERY = 100
# ─────────────────────────────────────────────────────────────────────


def get_answer_log_prob(model, problem):
    story_text = " ".join(problem['story'])
    prompt = story_text + " " + problem['question']
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
    answer_tokens = model.to_tokens(
        " " + problem['answer'], prepend_bos=False
    )
    return log_probs[answer_tokens[0, 0].item()].item()


def extract_all_layers(model, sentence):
    """
    One forward pass, return activation at final token for EVERY layer.
    Returns array of shape (n_layers, 768).
    """
    tokens = model.to_tokens(sentence)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return np.stack([
        cache['resid_post', L][0, -1, :].cpu().numpy()
        for L in range(N_LAYERS)
    ])


def cache_path(task_file):
    base = task_file.replace('.txt', '')
    return os.path.join(CACHE_DIR, f"{base}_alllayers.npz")


def load_partial(path):
    """Resume from existing cache. Returns (q_acts, supp_groups, lps)."""
    if not os.path.exists(path):
        return [], [], []
    d = np.load(path)
    # stored as (n_layers, N, 768) -> list of (n_layers, 768)
    q_acts = [d['question_acts'][:, i, :]
              for i in range(d['question_acts'].shape[1])]
    lengths = d['supp_lengths']
    flat = d['supp_acts_flat']       # (n_layers, total, 768)
    supp_groups, idx = [], 0
    for n in lengths:
        n = int(n)
        supp_groups.append([
            flat[:, idx + j, :] for j in range(n)
        ])
        idx += n
    return q_acts, supp_groups, list(d['log_probs'])


def save_cache(path, q_acts, supp_groups, lps):
    # q_acts: list of (n_layers, 768) -> (n_layers, N, 768)
    q_arr = np.stack(q_acts, axis=1)
    flat_list = [a for group in supp_groups for a in group]
    flat_arr = np.stack(flat_list, axis=1)   # (n_layers, total, 768)
    np.savez_compressed(
        path,
        question_acts=q_arr.astype(np.float32),
        supp_acts_flat=flat_arr.astype(np.float32),
        supp_lengths=np.array([len(g) for g in supp_groups]),
        log_probs=np.array(lps),
    )


def cache_task(model, task_file):
    path = cache_path(task_file)
    q_acts, supp_groups, lps = load_partial(path)
    done = len(lps)

    problems = parse_babi_file(
        os.path.join(DATA_DIR, task_file)
    )[:N_PROBLEMS]

    print(f"\n{'='*64}")
    print(f"CACHING ALL LAYERS: {task_file}")
    print('='*64)
    print(f"  Target:         {len(problems)} problems x "
          f"{N_LAYERS} layers")
    print(f"  Already cached: {done}")

    if done >= len(problems):
        print("  Already complete. Skipping.")
        return

    t0 = time.time()
    for i in range(done, len(problems)):
        p = problems[i]
        lps.append(get_answer_log_prob(model, p))
        q_acts.append(extract_all_layers(model, p['question']))
        supp_groups.append([
            extract_all_layers(model, s)
            for s in p['supporting_sentences']
        ])

        n_done = i + 1
        if n_done % SAVE_EVERY == 0 or n_done == len(problems):
            save_cache(path, q_acts, supp_groups, lps)
            elapsed = time.time() - t0
            rate = (n_done - done) / elapsed
            remaining = (len(problems) - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{len(problems)}] saved. "
                  f"{rate:.2f} prob/s, ~{remaining/60:.1f} min left")

    print(f"  Done in {(time.time()-t0)/60:.1f} min.")
    print(f"  Size: {os.path.getsize(path)/1e6:.1f} MB")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("="*64)
    print("ALL-LAYER ACTIVATION CACHE BUILDER")
    print("="*64)
    print(f"{N_LAYERS} layers, {N_PROBLEMS} problems/task, "
          f"tasks {TASKS}")
    print("Resumable. Same compute as single-layer; more disk.")

    print("\nLoading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    print("Model loaded.")

    for task_file in TASKS:
        cache_task(model, task_file)

    print("\n" + "="*64)
    print("ALL-LAYER CACHING COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()