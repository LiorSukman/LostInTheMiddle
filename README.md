# Lost in the Middle: Serial Position Effects in LLMs

An empirical study investigating how document position in multi-document QA prompts affects retrieval accuracy in large language models, building on [Liu et al. (2023)](https://arxiv.org/abs/2307.03172).

## Key Findings

**Experiment 0 — Replication:** The U-shaped accuracy curve from Liu et al. is replicated with GPT-4o-mini on a custom Wikipedia 2024-2025 dataset. Documents at the beginning (80%) and end (82%) of the context are retrieved significantly more accurately than those in the middle (70-72%). Cochran Q test: p = 0.0001.

**Experiment 1 — Preamble Length:** Adding neutral text before the document list *improves* accuracy and *eliminates* the positional bias — the opposite of what the RoPE decay hypothesis predicts.

| Condition | Overall Accuracy | U-effect |
|---|---|---|
| No preamble | 77.6% | +9.4% |
| Short preamble (~130 words) | 83.4% | +4.1% |
| Long preamble (~1290 words) | 87.3% | +0.5% |

The Cochran Q test for positional effects goes from p = 0.00002 (no preamble) to p = 0.49 (long preamble) — a dose-dependent elimination of the Lost in the Middle phenomenon.

## Dataset

196 factoid questions built from Wikipedia articles about post-October 2023 events (Academy Awards, Olympics, Nobel Prizes, elections, etc.). Using events after the model's training cutoff eliminates parametric memory contamination — the model cannot have memorized these answers.

Each question includes a gold passage containing the answer and 29 semantically similar distractor passages ranked by sentence-transformer embeddings.

**Domain distribution:** Film/Awards (66), Sports (53), Music (28), Politics (19), Literature (15), Science (15)

## Experimental Design

- **Within-subjects:** Every question is tested at every position and condition
- **30 documents per prompt** (1 gold + 29 distractors)
- **Distractor order fixed per question** across all conditions — position is the only variable
- **Model:** GPT-4o-mini, temperature 0
- **Scoring:** Case-insensitive substring matching against answer aliases

## Project Structure

```
DatasetBuilder/
    pipeline/                   — 6-stage dataset construction pipeline
    build_30doc_dataset.py      — Script to generate 30-doc variant
    dataset/
        final_dataset.json      — 196 questions, 19 distractors (20-doc)
        final_dataset_30.json   — 196 questions, 29 distractors (30-doc)
        raw_articles/           — Source Wikipedia articles
        chunked_passages/       — Chunked passages with embeddings

src/litm/                       — Shared library
    api.py                      — OpenAI client wrapper with retry logic
    prompts.py                  — Prompt formatting (build_context, format_prompt)
    scoring.py                  — Answer evaluation (normalize, score)
    env.py                      — .env loader

notebooks/
    01_replication_20doc.ipynb   — Exp 0: 20-doc U-curve replication
    01_replication_30doc.ipynb   — Exp 0: 30-doc U-curve replication
    02_preamble_length.ipynb    — Exp 1: Preamble length × position

results/                        — Raw CSVs and plots (never regenerated)

data_misc/
    preambles.json              — Preamble texts for Exp 1
```

## Experiments

| Experiment | Status | Notebook |
|---|---|---|
| 0 — Replication (20-doc) | Done | `01_replication_20doc.ipynb` |
| 0 — Replication (30-doc) | Done | `01_replication_30doc.ipynb` |
| 1 — Preamble Length | Done | `02_preamble_length.ipynb` |
| 2 — Relative vs Absolute Position | Planned | — |
| 3 — Von Restorff Effect | Planned | — |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run a notebook
uv run jupyter notebook
```

## Statistical Methods

- **Cochran Q test** — omnibus test for positional effects (within-subjects)
- **McNemar test** with continuity correction — pairwise comparisons
- **Bonferroni correction** for multiple comparisons
- 95% confidence intervals via binomial standard error

## References

- Liu, N., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172). *Transactions of the Association for Computational Linguistics*.
