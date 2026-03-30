"""Stage 6 — Validate the final dataset."""

import json
from collections import Counter

from .config import DATASET_DIR


def answer_in_text(answer: str, text: str) -> bool:
    return answer.lower() in text.lower()


def run():
    """Run validation checks and produce a report."""
    print("=" * 60)
    print("STAGE 6 — Validating dataset")
    print("=" * 60)

    ds_path = DATASET_DIR / "final_dataset.json"
    if not ds_path.exists():
        raise FileNotFoundError("Run Stage 5 first — final_dataset.json not found.")

    dataset = json.loads(ds_path.read_text(encoding="utf-8"))

    issues = {
        "answer_presence_failures": [],
        "distractor_contamination": [],
        "distractor_count_errors": [],
        "passage_length_warnings": [],
        "duplicate_questions": [],
    }

    # Domain mapping
    domain_map = {}
    film_kw = ["academy", "oscar", "golden globe", "bafta", "film"]
    sports_kw = ["olympic", "euro 2024", "fifa", "australian open", "wimbledon", "tour de france"]
    music_kw = ["grammy", "music"]
    science_kw = ["nobel", "physics", "medicine", "chemistry"]
    politics_kw = ["election", "presidential", "politics"]
    lit_kw = ["booker", "pulitzer", "literature"]

    def classify_domain(article: str) -> str:
        a = article.lower()
        for kw in film_kw:
            if kw in a:
                return "Film/Awards"
        for kw in sports_kw:
            if kw in a:
                return "Sports"
        for kw in music_kw:
            if kw in a:
                return "Music"
        for kw in science_kw:
            if kw in a:
                return "Science"
        for kw in politics_kw:
            if kw in a:
                return "Politics"
        for kw in lit_kw:
            if kw in a:
                return "Literature"
        return "Other"

    domain_counts = Counter()
    questions_seen = {}

    for item in dataset:
        q = item["question"]
        answer = item["answer"]
        gold = item["gold_passage"]
        distractors = item["distractors"]
        article = item["gold_article"]

        # Check answer presence in gold passage
        if not answer_in_text(answer, gold):
            issues["answer_presence_failures"].append(item["id"])

        # Check distractor contamination
        for d in distractors:
            if answer_in_text(answer, d["text"]):
                issues["distractor_contamination"].append(
                    {"item_id": item["id"], "distractor_rank": d["rank"],
                     "distractor_chunk": d["chunk_id"]}
                )

        # Check distractor count
        if len(distractors) != 19:
            issues["distractor_count_errors"].append(
                {"item_id": item["id"], "count": len(distractors)}
            )

        # Check passage length
        wc = len(gold.split())
        if wc < 40 or wc > 150:
            issues["passage_length_warnings"].append(
                {"item_id": item["id"], "word_count": wc}
            )

        # Domain
        domain = classify_domain(article)
        domain_counts[domain] += 1

        # Duplicate detection
        q_lower = q.lower().strip()
        if q_lower in questions_seen:
            issues["duplicate_questions"].append(
                {"item_id": item["id"], "duplicate_of": questions_seen[q_lower]}
            )
        else:
            questions_seen[q_lower] = item["id"]

    total = len(dataset)
    failed = (
        len(issues["answer_presence_failures"])
        + len(issues["distractor_contamination"])
        + len(issues["distractor_count_errors"])
        + len(issues["duplicate_questions"])
    )
    passed = total - failed

    report = {
        "total_items": total,
        "passed_all_checks": passed,
        "answer_presence_failures": len(issues["answer_presence_failures"]),
        "distractor_contamination": len(issues["distractor_contamination"]),
        "distractor_count_errors": len(issues["distractor_count_errors"]),
        "passage_length_warnings": len(issues["passage_length_warnings"]),
        "duplicate_questions": len(issues["duplicate_questions"]),
        "domain_distribution": dict(domain_counts),
        "details": issues,
    }

    # Save report
    report_path = DATASET_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    print(f"""
Dataset Validation Report
=========================
Total items:              {total}
Passed all checks:        {passed}
Answer presence failures: {len(issues['answer_presence_failures'])}
Distractor contamination: {len(issues['distractor_contamination'])}
Distractor count errors:  {len(issues['distractor_count_errors'])}
Passage length warnings:  {len(issues['passage_length_warnings'])}
Duplicate questions:      {len(issues['duplicate_questions'])}
Domain distribution:""")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {domain:20s} {count:4d}  ({pct:.0f}%)")

    print(f"\nSaved to {report_path}")
    return report


if __name__ == "__main__":
    run()
