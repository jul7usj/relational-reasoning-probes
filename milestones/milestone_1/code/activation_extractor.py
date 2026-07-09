# milestone 1 — activation extractor
# purpose: extract residual stream activations at supporting sentence
#           positions for bAbI problems
# thursday 8

import torch
import os
import sys

# add the code directory to path so we can import babi_loader
sys.path.append(os.path.dirname(__file__))
from babi_loader import parse_babi_file
from transformer_lens import HookedTransformer

# ── configuration ────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
LAYER = 11          # which layer to extract from (final layer of GPT-2 small)
N_PROBLEMS = 10     # how many problems to process (keep small for first run)
# ─────────────────────────────────────────────────────────────────────

def extract_activations_for_sentence(model, sentence, layer=LAYER):
    """
    Run GPT-2 small on a single sentence.
    Return the residual stream activation at the final token position.
    Shape: (d_model,) = (768,)
    """
    tokens = model.to_tokens(sentence)  # shape: (1, seq_len)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # residual stream at final token of this sentence
    # cache['resid_post', layer] shape: (1, seq_len, d_model)
    activation = cache['resid_post', layer][0, -1, :]  # shape: (d_model,)
    return activation


def extract_problem_activations(model, problem, layer=LAYER):
    """
    For a single bAbI problem, extract activations at each
    supporting sentence position.

    Returns:
      supporting_activations: list of tensors, one per supporting sentence
                              each tensor shape: (768,)
      problem_metadata: dict with story, question, answer, supporting_sentences
    """
    supporting_activations = []

    for sentence in problem['supporting_sentences']:
        activation = extract_activations_for_sentence(model, sentence, layer)
        supporting_activations.append(activation)

    metadata = {
        'question':             problem['question'],
        'answer':               problem['answer'],
        'supporting_sentences': problem['supporting_sentences'],
        'n_supporting':         len(problem['supporting_sentences'])
    }

    return supporting_activations, metadata


def main():
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    print("Model loaded.\n")

    # load task 1 only for first run
    print("Loading bAbI task 1...")
    task1 = parse_babi_file(os.path.join(DATA_DIR, 'qa1_train.txt'))
    print(f"Loaded {len(task1)} problems. Processing first {N_PROBLEMS}.\n")

    results = []

    for i, problem in enumerate(task1[:N_PROBLEMS]):
        activations, metadata = extract_problem_activations(model, problem)

        results.append({
            'activations': activations,
            'metadata':    metadata
        })

        print(f"Problem {i+1}:")
        print(f"  Question:   {metadata['question']}")
        print(f"  Answer:     {metadata['answer']}")
        print(f"  Supporting: {metadata['supporting_sentences']}")
        print(f"  Activation shapes: "
              f"{[list(a.shape) for a in activations]}")
        print()

    # sanity check on first result
    first_activation = results[0]['activations'][0]
    print("── Sanity check ──────────────────────────────────────")
    print(f"First supporting sentence activation shape: "
          f"{first_activation.shape}")
    print(f"Expected:                                   "
          f"torch.Size([768])")
    print(f"Match: {first_activation.shape == torch.Size([768])}")
    print(f"Activation norm (should be non-zero): "
          f"{first_activation.norm():.4f}")
    print("──────────────────────────────────────────────────────")
    print(f"\nActivation extraction confirmed.")
    print(f"Next step: build relational and coordinate feature "
          f"vectors from these activations.")

    return results


if __name__ == "__main__":
    main()