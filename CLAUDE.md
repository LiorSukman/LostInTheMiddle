# LITM Paper — Serial Position Effects in LLMs

## Project Overview

Empirical study investigating the "Lost in the Middle" phenomenon in LLMs — how document position in multi-document QA prompts affects retrieval accuracy. Built on Liu et al. (2023), extending with preamble length, relative vs absolute position, and Von Restorff distinctiveness experiments.

**Model:** GPT-4o-mini via OpenAI API
**Dataset:** 196 questions from Wikipedia 2024-2025 events (post-training-cutoff to avoid parametric memory contamination)

## Key Files

- `research_plan.txt` — Full experimental design, hypotheses, results, and statistical methods
- `DatasetBuilder/dataset/final_dataset_30.json` — Primary dataset (196 questions, 29 distractors each)
- `DatasetBuilder/dataset/final_dataset.json` — 20-doc variant (kept for reference)
- `data_misc/preambles.json` — Preamble texts for Experiment 1 (all 4 levels stored)
- `src/litm/` — Shared library (API client, prompt formatting, scoring)
- `notebooks/` — Experiment notebooks (one per experiment)
- `results/` — Raw CSV results and plots (never re-run inference to regenerate)

## Code Conventions

- All experiments run in Jupyter notebooks under `notebooks/`
- Shared code lives in `src/litm/` and is imported via `sys.path.insert`
- Results are saved as CSV immediately after experiment runs and never overwritten
- Plots saved as PNG to `results/` at 150 DPI
- Use `uv run` to execute Python (project uses uv for dependency management)

## Experimental Design Principles

- **Within-subjects design**: every question is tested at every position and condition — this is non-negotiable as it controls for question difficulty
- **Distractor order fixed per question**: shuffle distractors once per question (seeded by question index), then insert gold doc at each position into that fixed sequence. This ensures position is the only variable
- **Checkpointing**: long-running experiments checkpoint to CSV every 50 questions and support resume via a `completed` set
- **Temperature 0, max_tokens 50** for all API calls

## Statistical Methods

- **Cochran Q test**: primary omnibus test for within-subjects binary outcomes across k conditions
- **McNemar test** (with continuity correction): pairwise comparisons between specific conditions
- **Bonferroni correction**: applied when doing multiple pairwise McNemar tests
- **Point-biserial correlation**: for continuous × binary relationships (e.g., prompt length × accuracy)
- Report 95% CI using binomial standard error: `se = sqrt(p * (1-p) / n)`

## Scoring

Substring containment against answer aliases (case-insensitive, punctuation-stripped). Most errors are "I don't know" (genuine retrieval failures). Occasional near-misses treated as noise — they occur symmetrically across conditions.

## Current Status

- Experiment 0 (Replication): DONE — U-curve replicated with 30 docs (p=0.0001)
- Experiment 1 (Preamble Length): DONE — preamble eliminates U-curve (surprise finding)
- Experiment 2 (Relative vs Absolute): PLANNED
- Experiment 3 (Von Restorff): PLANNED

## Common Tasks

- To add a new experiment: create notebook `notebooks/0X_name.ipynb`, follow the pattern of existing notebooks (load data, run trials with checkpointing, plot, statistics, spot-check errors)
- To extend the dataset with more distractors: see `DatasetBuilder/build_30doc_dataset.py` as template
- To check API costs: use tiktoken (`tiktoken.encoding_for_model("gpt-4o-mini")`) for accurate token counts, not word-count heuristics
