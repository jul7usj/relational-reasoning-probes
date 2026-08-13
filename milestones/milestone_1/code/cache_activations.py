# milestone 1 — activation cache builder
# purpose: extract activations once, save to disk, so probe experiments
#           run in seconds instead of hours
# thursday 11
#
# saves per task, as .npz:
#   question_acts  (N, 768)      question context activation per problem
#   supp_acts_flat (total, 768)  all supporting activations concatenated
#   supp_lengths   (N,)          n supporting sentences per problem
#   log_probs      (N,)          log prob of correct answer token
#
# raw activations are saved (not derived features) so feature
# definitions can be changed later without re-extraction.
#
# resumable: re-running continues from where it stopped.

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
LAYER      = 11
N_PROBLEMS = 1000
TASKS      = ['qa1_train.txt', 'qa2_train.txt', 'qa3_train.txt']
SAVE_EVERY = 100      # checkpoint frequency
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


def extract_activation(model, sentence, layer=LAYER):
    tokens = model.to_tokens(sentence)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache['resid_post', layer][0, -1, :].cpu().numpy()


def cache_path(task_file):
    base = task_file.replace('.txt', '')
    return os.path.join(CACHE_DIR, f"{base}_layer{LAYER}.npz")


def load_partial(path):
    """Load an existing cache if present. Returns lists, or empties."""
    if not os.path.exists(path):
        return [], [], [], []
    d = np.load(path)
    q_acts = list(d['question_acts'])
    lengths = list(d['supp_lengths'])
    flat = list(d['supp_acts_flat'])
    lps = list(d['log_probs'])
    # rebuild supporting activations grouped per problem
    supp_groups, idx = [], 0
    for n in lengths:
        supp_groups.append(flat[idx:idx + int(n)])
        idx += int(n)
    return q_acts, supp_groups, lps, lengths


def save_cache(path, q_acts, supp_groups, lps):
    flat = [a for group in supp_groups for a in group]
    np.savez_compressed(
        path,
        question_acts=np.array(q_acts),
        supp_acts_flat=np.array(flat),
        supp_lengths=np.array([len(g) for g in supp_groups]),
        log_probs=np.array(lps),
    )


def cache_task(model, task_file):
    path = cache_path(task_file)
    q_acts, supp_groups, lps, _ = load_partial(path)
    done = len(lps)

    problems = parse_babi_file(
        os.path.join(DATA_DIR, task_file)
    )[:N_PROBLEMS]

    print(f"\n{'='*60}")
    print(f"CACHING: {task_file}")
    print('='*60)
    print(f"  Target:        {len(problems)} problems")
    print(f"  Already cached: {done}")

    if done >= len(problems):
        print("  Already complete. Skipping.")
        return

    t0 = time.time()
    for i in range(done, len(problems)):
        p = problems[i]

        lps.append(get_answer_log_prob(model, p))
        q_acts.append(extract_activation(model, p['question']))
        supp_groups.append([
            extract_activation(model, s)
            for s in p['supporting_sentences']
        ])

        n_done = i + 1
        if n_done % SAVE_EVERY == 0 or n_done == len(problems):
            save_cache(path, q_acts, supp_groups, lps)
            elapsed = time.time() - t0
            rate = (n_done - done) / elapsed
            remaining = (len(problems) - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{len(problems)}] saved. "
                  f"{rate:.2f} prob/s, "
                  f"~{remaining/60:.1f} min remaining")

    print(f"  Done in {(time.time()-t0)/60:.1f} min.")
    print(f"  Cache: {path}")
    print(f"  Size:  {os.path.getsize(path)/1e6:.1f} MB")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("="*60)
    print("ACTIVATION CACHE BUILDER")
    print("="*60)
    print(f"Layer {LAYER}, {N_PROBLEMS} problems/task, tasks {TASKS}")
    print("Resumable: re-run this script to continue if interrupted.")

    print("\nLoading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    print("Model loaded.")

    for task_file in TASKS:
        cache_task(model, task_file)

    print("\n" + "="*60)
    print("CACHING COMPLETE")
    print("="*60)
    print("Next: run probe_cached.py — experiments now take seconds.")


if __name__ == "__main__":
    main()