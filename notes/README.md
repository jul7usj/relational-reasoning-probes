# Notes

Working notes, decisions, and pointers. Less polished than the report
— written for the author, readable by visitors who want to understand
how the program developed.

Current files:

- `model_selection.md` — why GPT-2 small and bAbI tasks 1–3 were
  chosen for milestone 1, with the alternatives considered and
  rejected.
- `future_directions.md` — long-term research direction, logged out of
  the main report's scope. Includes the predictive coding connection
  (Rao & Ballard, 1999) as a convergent biological motivation.

## Environment

Python 3.12.0, torch 2.7.1+cpu, TransformerLens 3.4.0,
scikit-learn 1.9.0. CPU only, no GPU required. Packages installed in
the global environment, not a venv.

**Cache regeneration:** `.npz` files in `milestones/milestone_1/cache/`
are gitignored. Regenerate with
`python milestones/milestone_1/code/cache_all_layers.py` (~52 min CPU).

**Known noise:** a `ResourceTracker` `AttributeError` prints on
interpreter exit. Known Python 3.12 / multiprocess incompatibility,
harmless, ignore.