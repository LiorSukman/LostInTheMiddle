"""Stage 4c — Programmatic quality filters for Q&A pairs.

Stages 4a (closed-book) and 4b (oracle) are performed manually by Claude.
This module handles the programmatic filter (4c) and merges batch files.
"""

import json
import re
from pathlib import Path

from .config import DATASET_DIR


def merge_batches() -> list[dict]:
    """Merge all raw_qa_batch*.json files into one list."""
    all_pairs = []
    batch_files = sorted(DATASET_DIR.glob("raw_qa_batch*.json"))
    for bf in batch_files:
        pairs = json.loads(bf.read_text(encoding="utf-8"))
        all_pairs.extend(pairs)
        print(f"  Loaded {len(pairs)} pairs from {bf.name}")

    # Assign IDs
    for i, pair in enumerate(all_pairs):
        pair["id"] = f"qa_{i:04d}"
        pair["extraction_model"] = "claude-opus-4-6"
        pair["status"] = "raw"

    # Save merged
    out_path = DATASET_DIR / "raw_qa_pairs.json"
    out_path.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Total: {len(all_pairs)} raw pairs saved to {out_path.name}")
    return all_pairs


def filter_quality(pairs: list[dict]) -> tuple[list[dict], dict]:
    """Apply programmatic quality filters (Stage 4c).

    Returns (passing_pairs, stats_dict).
    """
    rejected = {"too_many_words": 0, "full_sentence": 0, "purely_numeric": 0, "tautology": 0}
    passing = []

    for pair in pairs:
        answer = pair["answer"].strip()
        question = pair["question"].strip().lower()

        # Check: answer > 5 words
        if len(answer.split()) > 5:
            rejected["too_many_words"] += 1
            continue

        # Check: answer is a full sentence (has subject + verb pattern)
        # Simple heuristic: if answer has > 3 words and contains common verbs
        if len(answer.split()) > 3:
            verb_pattern = r'\b(is|was|were|are|has|had|have|did|does|won|became|received)\b'
            if re.search(verb_pattern, answer, re.IGNORECASE):
                rejected["full_sentence"] += 1
                continue

        # Check: purely numeric with no context
        if re.match(r'^\d{4}$', answer):
            rejected["purely_numeric"] += 1
            continue

        # Check: tautology (answer appears verbatim in the question)
        if answer.lower() in question:
            rejected["tautology"] += 1
            continue

        passing.append(pair)

    return passing, rejected


def run():
    """Merge batches and apply quality filter."""
    print("=" * 60)
    print("STAGE 4c — Merging and quality-filtering Q&A pairs")
    print("=" * 60)

    pairs = merge_batches()
    passing, stats = filter_quality(pairs)

    print(f"\nQuality filter results:")
    print(f"  Input:           {len(pairs)}")
    print(f"  Too many words:  {stats['too_many_words']}")
    print(f"  Full sentences:  {stats['full_sentence']}")
    print(f"  Purely numeric:  {stats['purely_numeric']}")
    print(f"  Tautology:       {stats['tautology']}")
    print(f"  Passing:         {len(passing)}")

    return passing, stats


if __name__ == "__main__":
    run()
