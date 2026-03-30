"""Stage 4 — Full filtering pipeline.

4a: Closed-book contamination check (heuristic for GPT-4o-mini Oct 2023 cutoff)
4b: Oracle verification (answer must appear in source_sentence)
4c: Programmatic quality filters
"""

import json
import re
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Articles that are NOT exclusively post-Oct-2023 events
# These general articles may contain pre-cutoff facts
GENERAL_ARTICLES = {
    "Nobel Prize in Physics",
    "Nobel Prize in Physiology or Medicine",
}

# Known facts that GPT-4o-mini (Oct 2023 cutoff) could plausibly know
# These are conservative flags — we reject if there's any doubt
PRE_CUTOFF_KEYWORDS = [
    # People/entities well-known before Oct 2023
    # (the question itself might be answerable from general knowledge)
]


def answer_in_text(answer: str, text: str) -> bool:
    """Case-insensitive substring check."""
    return answer.lower().strip() in text.lower()


def run_filter_4a(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stage 4a: Closed-book contamination check.

    Since the model under test (GPT-4o-mini) has a knowledge cutoff of Oct 2023,
    and all our seed articles cover events after Oct 2023, most pairs should pass.

    We flag pairs from general articles (not year-specific) and pairs where
    the answer is a very well-known entity that might be guessable.
    """
    passing = []
    rejected = []

    for pair in pairs:
        article = pair["article"]
        question = pair["question"].lower()
        answer = pair["answer"].lower()

        reject_reason = None

        # Flag pairs from general (not year-specific) articles unless they
        # explicitly reference 2024/2025 events
        if article in GENERAL_ARTICLES:
            # Check if the question references 2024 or later
            if not re.search(r'202[4-9]|2030', question):
                reject_reason = "general_article_no_year_ref"

        # Flag purely numeric answers that could be guessed
        if answer.isdigit() and int(answer) < 100:
            # Small numbers are often guessable
            pass  # Allow, but note the risk

        # Flag answers that are just "Netflix", "The Bear", etc. — very well-known
        # These are OK if the question specifies a 2024/2025 context
        well_known_answers = {"netflix", "the bear"}
        if answer in well_known_answers and not re.search(r'202[4-9]', question):
            reject_reason = "well_known_answer_no_year"

        if reject_reason:
            pair["reject_reason_4a"] = reject_reason
            rejected.append(pair)
        else:
            passing.append(pair)

    return passing, rejected


def run_filter_4b(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stage 4b: Oracle verification.

    Check that the answer actually appears in the source_sentence.
    """
    passing = []
    rejected = []

    for pair in pairs:
        answer = pair["answer"]
        source = pair.get("source_sentence", "")

        if answer_in_text(answer, source):
            passing.append(pair)
        else:
            pair["reject_reason_4b"] = "answer_not_in_source"
            rejected.append(pair)

    return passing, rejected


def run_filter_4c(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stage 4c: Programmatic quality filters."""
    passing = []
    rejected = []

    for pair in pairs:
        answer = pair["answer"].strip()
        question = pair["question"].strip().lower()
        reject_reason = None

        # Answer > 5 words
        if len(answer.split()) > 5:
            reject_reason = "too_many_words"

        # Full sentence answer (>3 words with common verbs)
        elif len(answer.split()) > 3:
            verb_pattern = r'\b(is|was|were|are|has|had|have|did|does|won|became|received)\b'
            if re.search(verb_pattern, answer, re.IGNORECASE):
                reject_reason = "full_sentence"

        # Purely numeric year
        elif re.match(r'^\d{4}$', answer):
            reject_reason = "purely_numeric_year"

        # Tautology
        elif answer.lower() in question:
            reject_reason = "tautology"

        if reject_reason:
            pair["reject_reason_4c"] = reject_reason
            rejected.append(pair)
        else:
            passing.append(pair)

    return passing, rejected


def run():
    """Run all three filters sequentially."""
    print("=" * 60)
    print("STAGE 4 — Full Q&A filtering pipeline")
    print("=" * 60)

    # Load raw pairs
    raw_path = DATASET_DIR / "raw_qa_pairs.json"
    if not raw_path.exists():
        raise FileNotFoundError("raw_qa_pairs.json not found. Run extraction first.")

    all_pairs = json.loads(raw_path.read_text(encoding="utf-8"))
    print(f"Input: {len(all_pairs)} raw pairs\n")

    # Filter 4c first (cheapest)
    after_4c, rej_4c = run_filter_4c(all_pairs)
    print(f"Filter 4c (quality):     {len(all_pairs)} -> {len(after_4c)}  (rejected {len(rej_4c)})")
    for r in rej_4c:
        print(f"  REJECTED [{r.get('reject_reason_4c')}]: {r['question'][:70]}...")

    # Filter 4a (closed-book contamination)
    after_4a, rej_4a = run_filter_4a(after_4c)
    print(f"\nFilter 4a (closed-book): {len(after_4c)} -> {len(after_4a)}  (rejected {len(rej_4a)})")
    for r in rej_4a:
        print(f"  REJECTED [{r.get('reject_reason_4a')}]: {r['question'][:70]}...")

    # Filter 4b (oracle verification)
    after_4b, rej_4b = run_filter_4b(after_4a)
    print(f"\nFilter 4b (oracle):      {len(after_4a)} -> {len(after_4b)}  (rejected {len(rej_4b)})")
    for r in rej_4b:
        print(f"  REJECTED [{r.get('reject_reason_4b')}]: {r['question'][:70]}... (answer='{r['answer']}')")

    # Mark all passing pairs as verified
    for pair in after_4b:
        pair["passed_filters"] = True
        pair["status"] = "verified"

    # Save verified pairs
    out_path = DATASET_DIR / "verified_qa_pairs.json"
    out_path.write_text(json.dumps(after_4b, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save filter report
    report = {
        "total_extracted": len(all_pairs),
        "rejected_quality_4c": len(rej_4c),
        "rejected_closedbook_4a": len(rej_4a),
        "rejected_oracle_4b": len(rej_4b),
        "total_rejected": len(rej_4c) + len(rej_4a) + len(rej_4b),
        "verified_count": len(after_4b),
        "rejection_details_4c": [{"id": r["id"], "reason": r.get("reject_reason_4c"), "question": r["question"]} for r in rej_4c],
        "rejection_details_4a": [{"id": r["id"], "reason": r.get("reject_reason_4a"), "question": r["question"]} for r in rej_4a],
        "rejection_details_4b": [{"id": r["id"], "reason": r.get("reject_reason_4b"), "question": r["question"], "answer": r["answer"]} for r in rej_4b],
    }
    report_path = DATASET_DIR / "filter_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*40}")
    print(f"SUMMARY: {len(all_pairs)} -> {len(after_4b)} verified pairs")
    print(f"Saved to {out_path.name}")
    print(f"Report saved to {report_path.name}")

    return after_4b


if __name__ == "__main__":
    run()
