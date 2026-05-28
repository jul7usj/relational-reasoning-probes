# Relational Reasoning Probes

A research program probing whether reasoning in transformers can be analyzed
as motion through a relational state space rather than as next-token prediction
in a coordinate-embedded space.

**Status:** work in progress. Public from day one. Updated weekly.

## Thesis

The relational structure of activations across reasoning steps — not their
coordinate values — carries the reasoning load in transformer language models.
This claim is testable with a probe on small open-source models.

## Repository structure

- `report/` — the technical report. The roadmap and the writeup of the program.
- `milestones/` — executed work. Each subdirectory is one experimental milestone
  with its code, data, and results.
- `notes/` — scratch space: feasibility checks, model/benchmark selection,
  ongoing decisions.

## Current state

Phase 1 (scoping). Thesis and scope are being drafted in `report/v0_outline.md`.
Milestone 1 (the first probe) has not started yet — its design is being
specified in `milestones/milestone_1/`.

## Contact

Julien Rached Abboud — julienr.abboud@gmail.com
