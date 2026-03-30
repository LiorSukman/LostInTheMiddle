"""Expand answer_aliases for all dataset items.

Handles:
- Number words <-> digits (five <-> 5)
- Currency formatting variants ($560.3 million <-> 560.3 million)
- Common name abbreviations (first + last -> last only)
- Percentage variants (49.8% <-> 49.8 percent)
"""

import json
import re
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Bidirectional word <-> digit mapping
WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}
DIGIT_TO_WORD = {v: k for k, v in WORD_TO_DIGIT.items()}


def expand_number_aliases(answer: str) -> list[str]:
    """Generate numeric/textual variants of an answer."""
    aliases = set()
    lower = answer.lower().strip()

    # Case 1: answer is a single number word -> add digit form
    if lower in WORD_TO_DIGIT:
        aliases.add(WORD_TO_DIGIT[lower])

    # Case 2: answer is a bare integer -> add word form
    if lower.isdigit() and lower in DIGIT_TO_WORD:
        aliases.add(DIGIT_TO_WORD[lower])
        # Also add capitalized form
        aliases.add(DIGIT_TO_WORD[lower].capitalize())

    # Case 3: answer contains a number word as part of a descriptive phrase
    # e.g. "five awards" -> "5 awards", but NOT "Inside Out 2" style titles
    # Only apply if the answer doesn't look like a proper title (has lowercase words)
    tokens = lower.split()
    if len(tokens) >= 2 and not answer[0].isupper():
        for word, digit in WORD_TO_DIGIT.items():
            if word in tokens:
                variant = lower.replace(word, digit)
                aliases.add(variant)

    # Case 4: answer contains digits in a descriptive phrase -> add word variant
    # e.g. "5 minutes and 37 seconds" -> "five minutes and 37 seconds"
    # Skip titled answers like "Inside Out 2", "Terrifier 3"
    if len(tokens) >= 2 and not answer[0].isupper():
        new_tokens = list(tokens)
        changed = False
        for i, tok in enumerate(new_tokens):
            if tok in DIGIT_TO_WORD:
                new_tokens[i] = DIGIT_TO_WORD[tok]
                changed = True
        if changed:
            aliases.add(" ".join(new_tokens))

    return list(aliases)


def expand_currency_aliases(answer: str) -> list[str]:
    """Generate currency formatting variants."""
    aliases = set()

    # Strip currency symbols: $, A$, £, EUR, etc.
    # "$560.3 million" -> "560.3 million"
    currency_match = re.match(r'^(?:A?\$|£|€)\s*(.+)$', answer)
    if currency_match:
        aliases.add(currency_match.group(1))

    # "19.69 million" -> "19.69M"
    million_match = re.match(r'^([\d.,]+)\s*million$', answer, re.IGNORECASE)
    if million_match:
        aliases.add(f"{million_match.group(1)}M")
        aliases.add(f"{million_match.group(1)} Million")

    # Handle comma-less vs comma numbers: "96,500,000" <-> "96500000"
    if ',' in answer:
        aliases.add(answer.replace(',', ''))

    return list(aliases)


def expand_percentage_aliases(answer: str) -> list[str]:
    """Generate percentage variants."""
    aliases = set()

    # "49.8%" -> "49.8 percent", "49.8 %"
    pct_match = re.match(r'^([\d.]+)%$', answer.strip())
    if pct_match:
        num = pct_match.group(1)
        aliases.add(f"{num} percent")
        aliases.add(f"{num} %")
        aliases.add(f"{num}%")

    # "49.8 percent" -> "49.8%"
    pct_word_match = re.match(r'^([\d.]+)\s*percent$', answer.strip(), re.IGNORECASE)
    if pct_word_match:
        num = pct_word_match.group(1)
        aliases.add(f"{num}%")

    return list(aliases)


def expand_name_aliases(answer: str) -> list[str]:
    """Generate name variants (last name only, initials, etc.)."""
    aliases = set()
    parts = answer.split()

    # Common words that look like names but aren't — skip these as last-name aliases
    COMMON_WORDS = {
        "Me", "Us", "Day", "Cup", "Woman", "Man", "Door", "King", "Casting",
        "Generation", "Studio", "Pictures", "University", "Wilderness", "Madrid",
        "Atlantic", "Film", "Award", "Prize", "Out", "Room",
    }

    # "Conan O'Brien" -> "O'Brien"
    # "Mikey Madison" -> "Madison"
    # Only for 2-3 word names where all parts are capitalized and last name is
    # long enough to be a real surname (>= 4 chars) and not a common word
    if 2 <= len(parts) <= 3 and all(p[0].isupper() for p in parts if p[0].isalpha()):
        last = parts[-1]
        if len(last) >= 4 and last not in COMMON_WORDS:
            aliases.add(last)

    # "Santiago González and Giuliana Olmos" -> "González and Olmos"
    if " and " in answer:
        halves = answer.split(" and ")
        if len(halves) == 2:
            last_names = []
            for half in halves:
                name_parts = half.strip().split()
                if len(name_parts) >= 2:
                    last = name_parts[-1]
                    if len(last) >= 4 and last not in COMMON_WORDS:
                        last_names.append(last)
            if len(last_names) == 2:
                aliases.add(f"{last_names[0]} and {last_names[1]}")

    return list(aliases)


def expand_misc_aliases(answer: str) -> list[str]:
    """Handle miscellaneous formatting variants."""
    aliases = set()

    # "136 pages" -> "136"
    pages_match = re.match(r'^(\d+)\s*pages?$', answer, re.IGNORECASE)
    if pages_match:
        aliases.add(pages_match.group(1))

    # "41 times" -> "41"
    times_match = re.match(r'^(\d+)\s*times?$', answer, re.IGNORECASE)
    if times_match:
        aliases.add(times_match.group(1))

    # Lowercase variant — useful for scoring since models may not match case
    if answer != answer.lower():
        aliases.add(answer.lower())

    return list(aliases)


def expand_all_aliases(answer: str) -> list[str]:
    """Generate all alias variants for an answer."""
    all_aliases = set()
    all_aliases.add(answer)  # always include the original

    all_aliases.update(expand_number_aliases(answer))
    all_aliases.update(expand_currency_aliases(answer))
    all_aliases.update(expand_percentage_aliases(answer))
    all_aliases.update(expand_name_aliases(answer))
    all_aliases.update(expand_misc_aliases(answer))

    # Remove empty strings and the original if it's the only one
    all_aliases.discard("")
    return sorted(all_aliases)


def run():
    """Expand aliases in the final dataset."""
    ds_path = DATASET_DIR / "final_dataset.json"
    dataset = json.loads(ds_path.read_text(encoding="utf-8"))

    expanded_count = 0
    for item in dataset:
        original = item["answer"]
        aliases = expand_all_aliases(original)
        if len(aliases) > 1:
            expanded_count += 1
            old = item["answer_aliases"]
            item["answer_aliases"] = aliases
            print(f"  {item['id']:12s} {original:30s} -> {aliases}")
        else:
            item["answer_aliases"] = aliases

    ds_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nExpanded aliases for {expanded_count} / {len(dataset)} items.")
    print(f"Saved to {ds_path}")


if __name__ == "__main__":
    run()
