# Dataset Builder Pipeline — Rerun Guide

This document describes how to reproduce the LITM research dataset from scratch. The pipeline builds a multi-document QA dataset for studying positional bias in LLMs, immune to parametric memory contamination.

## Prerequisites

- Python 3.13+ with `uv` package manager
- Dependencies installed via `uv sync` from the project root
- An active Claude Code session (Stages 3 and 4a/4b require Claude directly)

## Pipeline Overview

```
Stage 1: Harvest       Wikipedia → raw_articles/             [Python script]
Stage 2: Chunk         raw_articles/ → chunked_passages/     [Python script]
Stage 3: Extract QA    articles → raw_qa_batch*.json         [Claude in chat]
Stage 4: Filter QA     raw_qa_batch*.json → verified_qa_pairs.json  [Python + Claude]
Stage 5: Distractors   verified_qa_pairs.json → final_dataset.json  [Python script]
Stage 6: Validate      final_dataset.json → validation_report.json  [Python script]
```

## Step-by-Step Instructions

### Stage 1 — Harvest Wikipedia Articles

**Automated.** Fetches 25 seed articles from the Wikipedia REST API.

```bash
cd D:\LITM_paper
uv run python -m DatasetBuilder.run_stage 1
```

- Fetches articles listed in `pipeline/config.py` → `SEED_ARTICLES`
- Follows Wikipedia redirects automatically
- Some titles don't match Wikipedia exactly. Six titles required manual correction during the initial run (see below). If rerunning, the corrected titles are already fetched by a follow-up script call:
  - `"2024 Golden Globe Awards"` → `"81st Golden Globe Awards"`
  - `"2024 BAFTA Awards"` → `"77th British Academy Film Awards"`
  - `"2025 BAFTA Awards"` → `"78th British Academy Film Awards"`
  - `"2024 Nobel Prize in Physics"` → `"Nobel Prize in Physics"` (general article)
  - `"2024 Nobel Prize in Medicine"` → `"Nobel Prize in Physiology or Medicine"` (general article)
  - `"2024 Nobel Prize in Chemistry"` → `"2024 Nobel Prizes"` (combined article)

To re-fetch the corrected titles, run the supplemental fetch from the chat history or add the corrected titles to `SEED_ARTICLES` in `config.py`.

**Output:** `dataset/raw_articles/` (25 `.txt` files + `metadata.json`)

---

### Stage 2 — Chunk Articles

**Automated.** Splits articles into 80–115 word passages.

```bash
uv run python -m DatasetBuilder.run_stage 2
```

- Splits on paragraph boundaries; splits mid-paragraph only above 200 words
- Prepends `[Section: Name]` prefix to each chunk for context
- Discards chunks < 30 words and list-only chunks

**Output:** `dataset/chunked_passages/` (25 per-article `.json` files + `all_chunks.json`)

---

### Stage 3 — Extract Q&A Pairs (Claude in Chat)

**Manual — performed by Claude in the conversation.** This is the step that replaces GPT-4o-mini API calls.

Claude reads the raw articles and extracts 10 factoid Q&A pairs per article. This was done by launching 4 parallel Claude agents, each handling a batch of articles:

- **Batch 1:** Film & Awards articles (6 articles → 60 pairs)
- **Batch 2:** Music & Film overview articles (5 articles → 50 pairs)
- **Batch 3:** Sports articles (6 articles → 60 pairs)
- **Batch 4:** Science, Politics & Literature (8 articles → 67 pairs)

**To rerun this step**, ask Claude to read the articles and extract QA pairs following the requirements in `spec.txt` (Stage 3 section). The prompt should specify:
1. Answer must be 1–5 words, appearing verbatim in the article
2. Question must be unanswerable without the article (post-October 2023 facts only)
3. Include the exact `source_sentence` from the article
4. Output as a JSON array to `dataset/raw_qa_batch{N}.json`

**Output:** `dataset/raw_qa_batch1.json` through `raw_qa_batch4.json`

---

### Stage 4 — Filter Q&A Pairs

**Mostly automated.** Combines the three spec filters into one script.

```bash
uv run python -m DatasetBuilder.run_stage 4
```

This runs three filters sequentially:

- **4c (Quality):** Programmatic — rejects answers > 5 words, full sentences, bare years, tautologies
- **4a (Closed-book contamination):** Heuristic — rejects pairs from general (non-year-specific) articles and pairs with well-known answers that GPT-4o-mini might guess without context. Since all seed articles cover events after GPT-4o-mini's October 2023 cutoff, this filter is conservative.
- **4b (Oracle verification):** Programmatic — rejects pairs where the answer string doesn't appear in the source_sentence (ensuring the gold passage is sufficient)

**Note on 4a:** The original spec calls for sending each question to GPT-4o-mini closed-book. Since we're not using any LLM API, this is approximated by heuristic rules. All articles are post-October 2023, so contamination risk is inherently low. If more rigorous filtering is needed, this step could be enhanced by having Claude attempt closed-book answers in chat.

**Output:** `dataset/raw_qa_pairs.json`, `dataset/verified_qa_pairs.json`, `dataset/filter_report.json`

---

### Stage 5 — Build Distractor Sets

**Automated.** Constructs 19-distractor sets for each verified Q&A pair.

```bash
uv run python -m DatasetBuilder.run_stage 5
```

- Uses `sentence-transformers/all-MiniLM-L6-v2` to rank candidate chunks by cosine similarity to the question
- Excludes any chunk containing the answer string (case-insensitive)
- Applies small similarity boosts for same-article (+0.05) and same-domain (+0.02) chunks
- Three-tier gold passage matching: (1) chunk with best source_sentence overlap, (2) any chunk containing the answer, (3) passage extracted directly from raw article text
- Pairs that can't produce a valid gold passage are skipped

**Output:** `dataset/final_dataset.json` (primary output consumed by the research notebook)

---

### Stage 6 — Validate

**Automated.** Runs integrity checks and produces a report.

```bash
uv run python -m DatasetBuilder.run_stage 6
```

Checks: answer presence in gold passages, distractor cleanliness, distractor count, passage length, duplicates, and domain distribution.

**Output:** `dataset/validation_report.json`

---

## Full Pipeline (All Stages)

```bash
# Stage 1 & 2: automated
uv run python -m DatasetBuilder.run_stage 1
uv run python -m DatasetBuilder.run_stage 2

# Stage 3: ask Claude to extract QA pairs (see instructions above)
# Stage 4: automated (after Stage 3 outputs exist)
uv run python -m DatasetBuilder.run_stage 4

# Stage 5 & 6: automated
uv run python -m DatasetBuilder.run_stage 5
uv run python -m DatasetBuilder.run_stage 6
```

## Run Statistics (Initial Build)

| Metric | Value |
|---|---|
| Articles harvested | 25 (19 direct + 6 corrected titles) |
| Total chunks | 573 |
| Raw QA pairs extracted | 237 |
| Rejected by quality filter (4c) | 6 |
| Rejected by closed-book filter (4a) | 13 |
| Rejected by oracle filter (4b) | 16 |
| Verified QA pairs | 202 |
| Final dataset items (with distractors) | 196 |
| Items lost at gold-passage matching | 6 |
