# Notes

Scratch space for the program. Feasibility checks, model and benchmark
selection rationale, hardware constraints, decisions made and rejected,
forward-looking pointers. Less polished than the report — written for the
author, readable by visitors who want to understand the program's
working state.

Current files:

- `future_directions.md` — long-term research direction logged out of
  the main report's scope.
  **Thursday 5 environment status:** torch 2.7.1+cpu and TransformerLens
3.4.0 confirmed working. HookedTransformer imports cleanly. Environment
is fully operational for Thursday 6 probe work. ResourceTracker warning
on exit is a known Python 3.12 / multiprocess noise — ignore permanently.