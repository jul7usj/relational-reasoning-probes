# milestone 1 — probe setup
# purpose: confirm GPT-2 small loads and activation extraction works
# thursday 6 — first run

from transformer_lens import HookedTransformer
import torch

print("Loading GPT-2 small via TransformerLens...")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("Model loaded.")

# run a single forward pass with activation caching
test_input = "The cat sat on the mat. Where did the cat sit?"
print(f"\nTest input: {test_input}")

with torch.no_grad():
    logits, cache = model.run_with_cache(test_input)

# print the shapes of key activation tensors
print("\nActivation cache — key tensor shapes:")
print(f"  residual stream (final layer): {cache['resid_post', 11].shape}")
print(f"  residual stream (layer 0):     {cache['resid_post', 0].shape}")
print(f"  number of layers:              {model.cfg.n_layers}")
print(f"  hidden dimension (d_model):    {model.cfg.d_model}")

# confirm output
print("\nSetup confirmed. Pipeline works.")
print("Next step: load bAbI tasks 1-3 and extract activations per problem.")