# Dataset Builder — Code & Output Reference

## Code Structure

```
DatasetBuilder/
├── __init__.py                 # Package marker
├── run_stage.py                # CLI entry point
├── spec.txt                    # Original specification document
├── PIPELINE.md                 # Rerun guide (how to reproduce)
├── CODEBASE.md                 # This file (code & output reference)
├── pipeline/
│   ├── __init__.py
│   ├── config.py               # Shared configuration
│   ├── stage1_harvest.py       # Wikipedia article fetching
│   ├── stage2_chunk.py         # Article chunking
│   ├── stage4_filter.py        # Quality filter only (standalone, not used in main flow)
│   ├── stage4_full_filter.py   # Full 3-stage filter (4a + 4b + 4c)
│   ├── stage5_distractors.py   # Distractor set construction
│   └── stage6_validate.py      # Dataset validation
└── dataset/                    # All output files (see "Output Files" below)
```

---

## Code Files

### `run_stage.py`

CLI entry point for running pipeline stages. Usage: `uv run python -m DatasetBuilder.run_stage <N>` where N is the stage number (1, 2, 4, 5, or 6). Stage 3 is performed by Claude in the conversation, not by a script.

### `pipeline/config.py`

Shared constants used across all stages:
- **`DATASET_DIR`** — path to the `dataset/` output directory
- **`RAW_ARTICLES_DIR`** — path to `dataset/raw_articles/`
- **`CHUNKED_PASSAGES_DIR`** — path to `dataset/chunked_passages/`
- **`SEED_ARTICLES`** — list of 25 Wikipedia article titles to harvest

Also creates output directories on import.

### `pipeline/stage1_harvest.py`

Fetches Wikipedia articles via the MediaWiki API (`action=query`, `prop=extracts`, `explaintext=true`). Key functions:

- **`sanitize_title()`** — converts article titles to safe filenames (lowercase, underscores)
- **`fetch_article()`** — fetches a single article, follows redirects, returns `{title, text, length_chars}` or `None`
- **`run()`** — iterates over `SEED_ARTICLES`, saves each as a `.txt` file, writes `metadata.json`

Respects Wikipedia rate limits with a 0.5s delay between requests. Sets a `User-Agent` header per Wikipedia API guidelines.

### `pipeline/stage2_chunk.py`

Splits articles into passages of ~80–115 words. Key functions:

- **`parse_sections()`** — splits raw article text into `(section_name, text)` tuples by detecting `== Header ==` patterns
- **`split_paragraph()`** — breaks paragraphs exceeding 200 words at sentence boundaries
- **`is_list_chunk()`** — detects list-heavy chunks (>70% short lines) for exclusion
- **`chunk_article()`** — processes one article into chunk dicts with `[Section: Name]` prefix
- **`run()`** — chunks all articles, saves per-article JSON files and a flat `all_chunks.json` index

Discards chunks under 30 words and list-only chunks. Skips non-content sections (References, See Also, etc.).

### `pipeline/stage4_filter.py`

Standalone quality filter module (Stage 4c only). Contains `merge_batches()` and `filter_quality()`. **Not used in the main pipeline** — superseded by `stage4_full_filter.py`, which incorporates these checks.

### `pipeline/stage4_full_filter.py`

The full filtering pipeline combining all three Stage 4 sub-filters. Invoked via `run_stage.py` as stage 4. Key functions:

- **`run_filter_4c()`** — Programmatic quality filter. Rejects:
  - Answers with more than 5 words
  - Answers that are full sentences (>3 words with common verbs)
  - Answers that are bare 4-digit years (e.g. "2024")
  - Tautologies (answer appears in the question)

- **`run_filter_4a()`** — Closed-book contamination heuristic. Rejects:
  - Pairs from general articles (e.g. "Nobel Prize in Physics") whose questions don't reference 2024+
  - Pairs with well-known answers (e.g. "Netflix") whose questions lack a year-specific context

- **`run_filter_4b()`** — Oracle verification. Rejects pairs where the answer string doesn't appear (case-insensitive) in the `source_sentence`. This ensures the source passage is genuinely sufficient to answer the question.

- **`run()`** — applies filters in order 4c → 4a → 4b, saves `verified_qa_pairs.json` and `filter_report.json`

### `pipeline/stage5_distractors.py`

Constructs 19-distractor sets for each verified Q&A pair. Key functions:

- **`articles_match()`** — fuzzy article name matching that handles case differences, plurals, and substrings (e.g. "82nd Golden Globe Awards" matches "82nd Golden Globes")
- **`build_gold_passage()`** — finds the best chunk matching the source sentence via word overlap. Three-tier fallback: (1) chunk with highest overlap ≥ 0.4, (2) any same-article chunk containing the answer, (3) passage extracted from raw article text
- **`build_gold_from_raw_article()`** — last-resort fallback that extracts a passage around the answer directly from the raw article text
- **`get_related_articles()`** / **`get_domain()`** — maps articles to domain groups (film, sports, music, science, politics, literature) for distractor sourcing
- **`run()`** — for each pair: builds gold passage, filters candidate distractors (excluding any containing the answer), ranks by cosine similarity using `all-MiniLM-L6-v2`, selects top 19

Uses `DOMAIN_GROUPS` dict to define which articles are thematically related for distractor prioritization.

### `pipeline/stage6_validate.py`

Runs automated integrity checks on the final dataset. Checks:

- Answer presence: answer string appears in gold_passage
- Distractor cleanliness: answer string does NOT appear in any distractor
- Distractor count: exactly 19 per item
- Passage length: flags gold passages outside 40–150 words
- Duplicate detection: flags repeated questions
- Domain distribution: counts items by source domain

Produces a printed summary table and saves `validation_report.json`.

---

## What Was Done in Code vs. in Chat

| Aspect | Method | Details |
|---|---|---|
| Article fetching (Stage 1) | Python script | Automated Wikipedia API calls |
| Article chunking (Stage 2) | Python script | Automated text splitting |
| **QA extraction (Stage 3)** | **Claude in chat** | Claude read articles and extracted 10 QA pairs each. Done via 4 parallel agents processing article batches. No LLM API calls — Claude did it directly. |
| **Closed-book check (Stage 4a)** | **Heuristic in code** | Originally spec'd as GPT-4o-mini API calls. Replaced with heuristic rules since all articles are post-Oct-2023 (GPT-4o-mini's cutoff), making contamination inherently unlikely. |
| Oracle verification (Stage 4b) | Python script | Substring match — checks answer exists in source_sentence |
| Quality filter (Stage 4c) | Python script | Programmatic word count, pattern, and tautology checks |
| Distractor construction (Stage 5) | Python script | sentence-transformers embedding similarity ranking |
| Validation (Stage 6) | Python script | Automated integrity checks |
| Title correction (Stage 1 supplement) | Claude in chat | Searched Wikipedia API for correct titles of 6 missing articles, fetched them with a one-off Python snippet |

---

## Output Files

### `dataset/raw_articles/` (Stage 1)

- **25 `.txt` files** — plain text of each Wikipedia article, one per file. Filenames are sanitized article titles (e.g. `97th_academy_awards.txt`).
- **`metadata.json`** — array of objects, one per article:
  ```json
  { "title": "97th Academy Awards", "original_query": "97th Academy Awards",
    "filename": "97th_academy_awards.txt", "length_chars": 16321 }
  ```

### `dataset/chunked_passages/` (Stage 2)

- **25 per-article `.json` files** — array of chunk objects for each article
- **`all_chunks.json`** — flat array of all 573 chunks across all articles. Each chunk:
  ```json
  { "article": "97th Academy Awards", "section": "Best Picture",
    "chunk_id": "97th_academy_awards_014",
    "text": "[Section: Best Picture] Oppenheimer, directed by...",
    "word_count": 94 }
  ```

### `dataset/raw_qa_batch{1-4}.json` (Stage 3)

Four JSON files produced by Claude's parallel extraction agents. Each is a flat array of QA pair objects:
```json
{ "article": "97th Academy Awards",
  "question": "Who hosted the 97th Academy Awards ceremony?",
  "answer": "Conan O'Brien",
  "source_sentence": "Comedian Conan O'Brien hosted the show for the first time." }
```

Batch boundaries:
- **Batch 1** (60 pairs): 97th/98th Oscars, 81st/82nd Golden Globes, 77th/78th BAFTAs
- **Batch 2** (50 pairs): 66th/67th Grammys, 2024 in music, 2024/2025 in film
- **Batch 3** (60 pairs): Olympics, Euro 2024, FIFA, Australian Open, Wimbledon, Tour de France
- **Batch 4** (67 pairs): US election, politics, Booker/Pulitzer Prizes, literature, Nobel Prizes

### `dataset/raw_qa_pairs.json` (Stage 4 — intermediate)

Merged and ID-assigned version of all batches. 237 pairs total. Each entry adds:
- `"id"`: sequential ID like `"qa_0042"`
- `"extraction_model"`: `"claude-opus-4-6"`
- `"status"`: `"raw"`

### `dataset/verified_qa_pairs.json` (Stage 4 — output)

202 pairs that passed all three filters. Each entry adds:
- `"passed_filters": true`
- `"status": "verified"`

### `dataset/filter_report.json` (Stage 4 — diagnostic)

Summary of filter results:
```json
{ "total_extracted": 237,
  "rejected_quality_4c": 6, "rejected_closedbook_4a": 13, "rejected_oracle_4b": 16,
  "total_rejected": 35, "verified_count": 202,
  "rejection_details_4c": [...], "rejection_details_4a": [...], "rejection_details_4b": [...] }
```

Each rejection detail includes the pair's `id`, `reason`, `question`, and (for 4b) `answer`.

### `dataset/final_dataset.json` (Stage 5 — PRIMARY OUTPUT)

**This is the file consumed by the research notebook.** 196 dataset items, each structured as:

```json
{
  "id": "item_0042",
  "question": "Who hosted the 97th Academy Awards ceremony?",
  "answer": "Conan O'Brien",
  "answer_aliases": ["Conan O'Brien"],
  "gold_passage": "[Section: Introduction] The 97th Academy Awards ceremony...",
  "gold_article": "97th Academy Awards",
  "gold_chunk_id": "97th_academy_awards_000",
  "distractors": [
    {
      "rank": 1,
      "text": "[Section: Ceremony information] The Academy also announced...",
      "article": "97th Academy Awards",
      "chunk_id": "97th_academy_awards_011",
      "similarity_score": 0.6759
    },
    ... // 19 total distractors, ordered by decreasing similarity
  ]
}
```

Key fields:
- **`answer_aliases`**: list of acceptable answer variants (for scoring). Initially just `[answer]`; can be manually expanded for common abbreviations.
- **`gold_passage`**: the passage containing the answer. May come from a chunk (with `[Section:]` prefix) or from a raw-article extraction (without prefix).
- **`gold_chunk_id`**: the chunk ID if the passage came from a chunk; `null` if from raw-article fallback.
- **`distractors`**: 19 passages ranked by cosine similarity to the question (highest first), mimicking a RAG retrieval ranking. None contain the answer string.

### `dataset/validation_report.json` (Stage 6)

Validation results:
```json
{
  "total_items": 196,
  "passed_all_checks": 196,
  "answer_presence_failures": 0,
  "distractor_contamination": 0,
  "distractor_count_errors": 0,
  "passage_length_warnings": 36,
  "duplicate_questions": 0,
  "domain_distribution": {
    "Film/Awards": 66, "Sports": 53, "Music": 28,
    "Politics": 19, "Literature": 15, "Science": 15
  },
  "details": { ... }
}
```
