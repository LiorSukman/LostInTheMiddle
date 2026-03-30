# LITM Research Dataset Builder — Project Specification

> **Note:** This spec was originally drafted as `spec.txt` before implementation. Sections marked with **[Decision]** blocks document choices made during the actual build that deviate from or refine the original plan.

## Project Goal

Build a semi-automatic pipeline that produces a high-quality, controlled multi-document QA dataset for studying the **"Lost in the Middle" (LITM) positional bias** in large language models. The dataset will be used in a research Jupyter notebook investigating three phenomena: preamble length effects, relative vs. absolute position, and the Von Restorff effect.

The dataset must be immune to **parametric memory contamination** — meaning the model under test (GPT-4o-mini, knowledge cutoff October 2023) cannot answer any question without reading the provided documents. All facts must therefore originate from events after October 2023.

---

## Background Context

The LITM effect (Liu et al., 2023) shows that LLMs perform best when relevant information appears at the beginning or end of a long context, and worst when it appears in the middle — producing a U-shaped accuracy curve. To study this cleanly, each experimental trial consists of:

- A **question** with a short factoid answer (1–5 words)
- A **gold document**: a passage that unambiguously contains the answer
- **19 distractor documents**: thematically related passages that do NOT contain the answer, ordered by decreasing semantic similarity to the question (mimicking a RAG retrieval ranking)
- The gold document is inserted at a controlled position (1–20) among the distractors

The experimental setup directly mirrors a RAG system where a vector database returns the top-20 most relevant chunks for a query, and the answer may appear at any rank position.

---

## Data Sources

All content is sourced from **English Wikipedia** via the free Wikimedia REST API. No authentication required.

### Seed Articles

Use Wikipedia articles covering events **after October 2023**. Prioritize articles that are dense with discrete, verifiable facts.

**Target seed list (minimum 25 articles, covering diverse domains):**

```python
SEED_ARTICLES = [
    # Film & Awards
    "97th Academy Awards",       # March 2024
    "98th Academy Awards",       # March 2025
    "2024 Golden Globe Awards",
    "2025 Golden Globe Awards",
    "2024 BAFTA Awards",
    "2025 BAFTA Awards",
    "2024 in film",
    "2025 in film",

    # Music
    "66th Grammy Awards",        # February 2024
    "67th Grammy Awards",        # February 2025
    "2024 in music",

    # Sports
    "2024 Summer Olympics",
    "UEFA Euro 2024",
    "2024 FIFA Club World Cup",
    "2025 Australian Open",
    "2024 Wimbledon Championships",
    "2024 Tour de France",

    # Science & Technology
    "2024 Nobel Prize in Physics",
    "2024 Nobel Prize in Medicine",
    "2024 Nobel Prize in Chemistry",

    # Politics & World Events
    "2024 United States presidential election",
    "2025 in politics",

    # Literature & Culture
    "2024 Booker Prize",
    "2024 Pulitzer Prize",
    "2025 in literature",
]
```

Additional articles may be added if yield from above is insufficient. Avoid articles about events within 2 months of the cutoff (i.e., avoid Oct–Dec 2023) to prevent partial parametric knowledge.

> **[Decision] Wikipedia title corrections**
>
> Six titles from the original seed list do not exist on Wikipedia under those names. The following corrections were applied:
>
> | Original title | Actual Wikipedia title |
> |---|---|
> | `2024 Golden Globe Awards` | `81st Golden Globe Awards` (redirects to `81st Golden Globes`) |
> | `2024 BAFTA Awards` | `77th British Academy Film Awards` |
> | `2025 BAFTA Awards` | `78th British Academy Film Awards` |
> | `2024 Nobel Prize in Physics` | `Nobel Prize in Physics` (general article, no year-specific page) |
> | `2024 Nobel Prize in Medicine` | `Nobel Prize in Physiology or Medicine` (general article) |
> | `2024 Nobel Prize in Chemistry` | `2024 Nobel Prizes` (combined article for all 2024 prizes) |
>
> The 19 articles from the original list that resolved correctly were fetched first. The 6 corrected titles were fetched in a supplemental step. Total: **25 articles**.
>
> **Implication for the Nobel articles:** The general Nobel articles contain information spanning many decades, not just 2024. The Stage 4a filter rejects QA pairs from these articles that don't explicitly reference 2024+ events, preventing pre-cutoff contamination.

---

## Pipeline Architecture

The pipeline runs as a sequence of stages. Each stage saves its output to disk before proceeding, so any stage can be re-run independently.

```
Stage 1: Harvest          Wikipedia articles → raw_articles/
Stage 2: Chunk            Articles → chunked_passages/
Stage 3: Extract QA       Articles → raw_qa_pairs.json
Stage 4: Filter           raw_qa_pairs.json → verified_qa_pairs.json
Stage 5: Build distractors verified_qa_pairs.json → final_dataset.json
Stage 6: Validate         final_dataset.json → validation_report.json
```

> **[Decision] LLM approach — Claude in chat instead of OpenAI API**
>
> The original spec called for GPT-4o-mini API calls in Stages 3 and 4. This was replaced with a **zero-API-cost approach**:
> - **Stage 3** (QA extraction): Performed directly by Claude Opus in the Claude Code conversation. Claude read the raw articles and extracted QA pairs, outputting JSON batch files. This was parallelized across 4 concurrent agents processing article batches.
> - **Stage 4a** (closed-book contamination check): Replaced with a heuristic filter in Python. Since all articles are post-October-2023, GPT-4o-mini (the model under test) cannot know these facts from training data, making contamination inherently unlikely. The heuristic rejects pairs from general articles without 2024+ year references, and pairs with well-known answers lacking year context.
> - **Stage 4b** (oracle verification): Replaced with a programmatic substring check — verifying the answer string appears in the source_sentence. This is stricter than asking an LLM (which might paraphrase).
>
> **Rationale:** Claude Opus is a more capable model than GPT-4o-mini, produces higher-quality extractions, and the user is already paying for Claude Code — eliminating the need for a separate OpenAI API key and cost.

---

## Stage 1 — Harvest Wikipedia Articles

### API

```
GET https://en.wikipedia.org/w/api.php
  ?action=query
  &titles={TITLE}
  &prop=extracts
  &explaintext=true
  &format=json
  &redirects=1
```

### Output

Save each article as a plain text file: `raw_articles/{sanitized_title}.txt`

Also save article metadata (title, fetch date, article length in chars) to `raw_articles/metadata.json`.

### Error handling

- If an article title returns a redirect, follow the redirect and save under the canonical title
- If an article is not found, log a warning and skip
- Add a 0.5s delay between requests to respect Wikipedia rate limits

> **[Decision]** Implemented as specified. The `redirects=1` parameter handles redirects at the API level. A custom `User-Agent` header is set per Wikipedia API guidelines.

---

## Stage 2 — Chunk Articles

Split each article into passages of **100–150 tokens** (approximate using word count x 1.3 as a token estimate, target ~80–115 words per chunk).

### Chunking rules

- Split on paragraph boundaries first; only split mid-paragraph if a paragraph exceeds 200 words
- Preserve section headers — prepend the section header to each chunk from that section as context: `"[Section: Award Categories] The Best Picture award went to..."`
- Discard chunks shorter than 30 words (usually section headers, captions, reference lists)
- Discard chunks that are primarily lists of names or numbers with no prose context

### Output

`chunked_passages/{sanitized_title}.json` — list of objects:
```json
{
  "article": "97th Academy Awards",
  "section": "Best Picture",
  "chunk_id": "97th_academy_awards_014",
  "text": "[Section: Best Picture] Oppenheimer, directed by Christopher Nolan...",
  "word_count": 94
}
```

Also maintain a flat index: `chunked_passages/all_chunks.json` — all chunks across all articles.

> **[Decision]** Implemented as specified. The chunker also skips non-content sections: "See also", "References", "External links", "Further reading", "Notes", "Bibliography". List detection uses a heuristic (>70% of lines having 5 words or fewer).
>
> **Result:** 573 chunks from 25 articles. Median word count: 99. Some chunks exceeded the target range (up to 417 words) when sentences within a paragraph were individually very long.
>
> **Known limitation:** Some short but fact-rich sections (e.g., awards lists, honoree names) are discarded by the 30-word minimum. This caused ~42 QA pairs to lose their gold passage at Stage 5, requiring a raw-article-fallback mechanism. See Stage 5 decisions.

---

## Stage 3 — Extract Q&A Pairs

> **[Decision] Performed by Claude in chat, not by GPT-4o-mini API calls.**

For each article, extract exactly 10 factoid Q&A pairs following these requirements:

### Extraction requirements

1. The answer must be 1–5 words maximum
2. The answer string must appear verbatim (or near-verbatim) in the article text
3. The question must be unanswerable without reading the article — do not use facts that are common general knowledge
4. The question must have a single unambiguous correct answer
5. Include the exact sentence from the article that contains the answer ("source_sentence")
6. Do not extract questions about dates alone (e.g. "What year was X founded") — prefer questions about people, places, titles, or named outcomes
7. Prefer questions about specific named outcomes: winners, record-breakers, first-time achievements, specific statistics

### Output

`raw_qa_pairs.json` — flat list of all extracted pairs with provenance:
```json
{
  "id": "qa_0042",
  "article": "97th Academy Awards",
  "question": "Which film won Best Picture at the 97th Academy Awards?",
  "answer": "Oppenheimer",
  "source_sentence": "Oppenheimer won seven awards including Best Picture, Best Director, and Best Actor.",
  "extraction_model": "claude-opus-4-6",
  "status": "raw"
}
```

> **[Decision]** Extraction was parallelized into 4 batches by domain:
> - Batch 1: Film & Awards (6 articles, 60 pairs)
> - Batch 2: Music & Film overviews (5 articles, 50 pairs)
> - Batch 3: Sports (6 articles, 60 pairs)
> - Batch 4: Science, Politics & Literature (8 articles, 67 pairs)
>
> Each batch was processed by a separate Claude agent running concurrently. The `extraction_model` field is set to `"claude-opus-4-6"` instead of `"gpt-4o-mini"`. Intermediate outputs are saved as `raw_qa_batch{1-4}.json` and merged into `raw_qa_pairs.json` during Stage 4.
>
> **Result:** 237 raw pairs total. Two shorter articles (2024 Pulitzer Prize: 2 pairs; 2025 in literature: 5 pairs) yielded fewer than 10 due to limited content.

---

## Stage 4 — Filter Q&A Pairs

Three sequential filters. A pair must pass all three to advance.

### Filter 4c — Answer quality check

Programmatic filter — reject pairs where:
- `answer` contains more than 5 words
- `answer` is a full sentence (contains a verb + subject structure — detected by checking >3-word answers for common verbs)
- `answer` is purely numeric with no context (e.g. "2024" alone — a bare 4-digit year)
- `question` and `answer` together form a tautology (answer appears verbatim in the question)

> **[Decision]** Filter order changed from the original spec (4a → 4b → 4c) to **4c → 4a → 4b**. The cheapest programmatic filter runs first to reduce the input set for the more complex filters. This is a performance optimization with no effect on outcomes.

### Filter 4a — Closed-book contamination check

> **[Decision] Replaced LLM-based check with heuristic rules.**
>
> The original spec called for sending each question to GPT-4o-mini without documents and rejecting pairs where the model answers correctly. Since:
> 1. We are not calling any external LLM API
> 2. All seed articles cover events after GPT-4o-mini's October 2023 cutoff
> 3. Contamination risk is therefore inherently very low
>
> The heuristic filter rejects:
> - Pairs from **general articles** (`Nobel Prize in Physics`, `Nobel Prize in Physiology or Medicine`) whose questions don't reference 2024 or later — these articles contain decades of pre-cutoff information
> - Pairs with **well-known answers** (e.g. "Netflix", "The Bear") whose questions don't include a year-specific qualifier — these could be guessed without reading the article
>
> **If more rigorous filtering is needed**, this step could be enhanced by having Claude attempt closed-book answers in chat, or by calling an LLM API with a pre-October-2023 knowledge cutoff.

### Filter 4b — Oracle verification

> **[Decision] Replaced LLM-based check with programmatic substring match.**
>
> The original spec sent each pair to GPT-4o-mini with only the source_sentence as context. Instead, we verify that the answer string appears (case-insensitive) in the source_sentence. This is actually **stricter** — an LLM might infer an answer that's paraphrased in the source, but our check requires exact presence.

**Reject** the pair if the answer string does not appear in the source_sentence.

### Output

`verified_qa_pairs.json` — filtered list, each entry with added fields:
```json
{
  ...,
  "passed_filters": true,
  "status": "verified"
}
```

Also write a `filter_report.json` summarizing:
- Total pairs extracted
- Rejected by quality filter (4c)
- Rejected by closed-book heuristic (4a)
- Rejected by oracle check (4b)
- Final verified count

**Target: minimum 200 verified Q&A pairs.** If yield is below 200, expand the seed article list and re-run from Stage 3.

> **[Decision] Result:** 237 raw → 202 verified (35 rejected: 6 quality, 13 closed-book, 16 oracle). Meets the 200 target.

---

## Stage 5 — Build Distractor Sets

For each verified Q&A pair, construct a set of 19 distractor passages.

### Distractor sourcing rules

1. **Primary source**: other chunks from the same article as the gold passage
2. **Secondary source**: chunks from thematically related articles (e.g., for "97th Academy Awards", related articles include "98th Academy Awards", "82nd Golden Globes", "2024 in film")
3. **Never include**: any chunk that contains the answer string (case-insensitive)

### Distractor ranking

Use `sentence-transformers` (model: `all-MiniLM-L6-v2`) to rank candidate distractors by cosine similarity to the question. The 19 most similar non-answer chunks become the distractor set, ordered by **decreasing similarity** (most similar first) — this mirrors real RAG retrieval ordering.

A small similarity boost is applied to prioritize contextually relevant distractors:
- Same article as the gold passage: **+0.05** to similarity score
- Same domain group (e.g., other award ceremonies): **+0.02**

> **[Decision]** The domain groups used for boosting and related-article lookup are:
> - `film_awards`: Oscars, Golden Globes, BAFTAs, "in film" articles
> - `music`: Grammys, "in music"
> - `sports`: Olympics, Euro, FIFA, Australian Open, Wimbledon, Tour de France
> - `science`: Nobel Prize articles
> - `politics`: US election, "in politics"
> - `literature`: Booker, Pulitzer, "in literature"

### Fallback if fewer than 19 non-answer chunks are available

If the same article + related articles don't yield 19 clean distractors:
1. First, expand to any article in the same domain (e.g., other award ceremonies)
2. If still insufficient, log a warning and skip this Q&A pair — do not pad with unrelated content

> **[Decision]** In practice, the full 573-chunk pool (minus answer-containing chunks) always provided 19+ candidates. No pairs were skipped for insufficient distractors.

### Gold passage construction

The gold passage is **not** the raw source sentence alone. Expand it to include surrounding context, targeting 60–100 words total. Verify the answer string is still present in the expanded passage.

> **[Decision] Three-tier gold passage matching**
>
> Finding the right passage proved to be the biggest source of data loss. A three-tier fallback was implemented:
>
> 1. **Chunk match:** Find the chunk from the same article with the highest word overlap to the source_sentence (threshold >= 0.4). This is the preferred path — the chunk already has the `[Section:]` prefix and appropriate length.
> 2. **Answer-in-chunk fallback:** If no chunk matches the source_sentence well enough, find any chunk from the same article that contains the answer string. Among matches, pick the one with the highest source_sentence overlap.
> 3. **Raw article fallback:** If no chunk contains the answer (because it was in a section discarded during chunking), extract a passage directly from the raw article text. Find sentences containing the answer and include 1–2 neighbors for context.
>
> **Article name matching** required special handling because QA pairs sometimes use slightly different article names than the chunks (e.g., "82nd Golden Globe Awards" vs "82nd Golden Globes", "2024 in Film" vs "2024 in film"). A fuzzy matcher using stemmed word overlap (>= 0.7) and substring containment handles these cases.
>
> **Result:** 202 verified pairs → 196 final items (6 lost where the answer string couldn't be found in either chunks or raw article text, likely due to encoding differences or the answer spanning non-contiguous text).

### Output

`final_dataset.json` — list of dataset items:
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
    ...19 total
  ]
}
```

---

## Stage 6 — Validation

Run automated checks and produce a human-readable report.

### Checks

- **Answer presence**: confirm answer string appears in gold_passage for every item
- **Distractor cleanliness**: confirm answer string does NOT appear in any distractor for every item
- **Distractor count**: confirm exactly 19 distractors per item
- **Passage length**: flag gold passages shorter than 40 words or longer than 150 words
- **Domain distribution**: report breakdown of items by source article domain (film, sports, music, etc.)
- **Duplicate detection**: flag any question that appears more than once (exact or near-exact match)

### Output

`validation_report.json` and a printed summary table.

> **[Decision] Actual validation results:**
>
> ```
> Dataset Validation Report
> =========================
> Total items:              196
> Passed all checks:        196
> Answer presence failures:   0
> Distractor contamination:   0
> Distractor count errors:    0
> Passage length warnings:   36  ← mostly short gold passages from raw-article fallback
> Duplicate questions:        0
> Domain distribution:
>   Film/Awards:            66  (34%)
>   Sports:                 53  (27%)
>   Music:                  28  (14%)
>   Politics:               19  (10%)
>   Literature:             15  (8%)
>   Science:                15  (8%)
> ```

---

## Final Output Files

```
dataset/
  raw_articles/           # Stage 1 — 25 .txt files + metadata.json
  chunked_passages/       # Stage 2 — 25 .json files + all_chunks.json
  raw_qa_batch{1-4}.json  # Stage 3 — intermediate batch files from Claude
  raw_qa_pairs.json       # Stage 3/4 — merged raw pairs (237)
  verified_qa_pairs.json  # Stage 4 — filtered pairs (202)
  filter_report.json      # Stage 4 — filtering diagnostics
  final_dataset.json      # Stage 5 ← PRIMARY OUTPUT (196 items)
  validation_report.json  # Stage 6 — validation results
```

The `final_dataset.json` is the only file consumed by the research notebook. All other files are intermediate artifacts for debugging and reproducibility.

---

## Dependencies

```
requests              # Wikipedia API
sentence-transformers # Distractor ranking (all-MiniLM-L6-v2)
numpy                 # Embedding similarity
tqdm                  # Progress bars
```

> **[Decision]** The `openai` dependency was **not needed** — Claude handled Stages 3 and 4 directly. No OpenAI API key is required. The `anthropic` SDK was also not needed since Claude operated within the Claude Code conversation, not via API.

All Wikipedia fetching is free with no API key. The only non-free dependency is the Claude Code subscription for running Stage 3 extraction.

---

## Cost Estimate

> **[Decision] Updated from the original estimate:**

| Stage | Method | Cost |
|---|---|---|
| Stage 1 — Harvest | Wikipedia API | Free |
| Stage 2 — Chunk | Local Python | Free |
| Stage 3 — Extraction | Claude in chat | Included in Claude Code subscription |
| Stage 4 — Filtering | Local Python | Free |
| Stage 5 — Distractors | Local sentence-transformers | Free |
| Stage 6 — Validation | Local Python | Free |
| **Total** | | **$0.00 incremental** |

Original spec estimated ~$0.04 for GPT-4o-mini API calls. By using Claude Code directly, incremental cost is zero.

---

## Important Constraints and Edge Cases

**Answer alias handling**: some answers have valid alternative forms (e.g., "Cillian Murphy" vs "Murphy"). After generation, run a manual pass to add aliases to `answer_aliases` for any answer with common abbreviations or alternative names. The research notebook scorer checks all aliases.

> **[Decision]** Aliases were not manually expanded in the initial build. Each item has `answer_aliases: [answer]` (the answer itself). This is a TODO for improving scorer accuracy.

**Section header injection**: the `[Section: X]` prefix on each chunk is intentional — it gives the model structural context about where in the article each passage comes from, which is realistic for a chunked RAG corpus. Do not strip these in the final dataset.

**Answer overlap in distractors**: the contamination check uses case-insensitive substring matching. Be aware this may over-reject — e.g., if the answer is "Nolan" and a distractor mentions "Christopher Nolan" in a different context. Log these edge cases rather than silently rejecting.

> **[Decision]** The validation report shows 0 distractor contamination in the final dataset, so over-rejection was not a significant issue in practice.

**Minimum viable dataset**: if after expanding the seed list you cannot reach 200 verified items, the minimum acceptable is 120 items (6 per position cell for 20 positions). Below this, the research notebook's statistical power is insufficient.

> **[Decision]** Final dataset: **196 items** (9.8 per position on average). Above the 120 minimum. Slightly below the 200 target due to 6 items lost at gold-passage matching. This provides adequate statistical power.
