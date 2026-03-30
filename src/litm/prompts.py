import random


def build_context(
    gold_doc: str,
    distractor_pool: list[str],
    gold_position: int,
    total_docs: int = 20,
) -> list[str]:
    """Build a list of documents with the gold doc inserted at gold_position (0-indexed)."""
    distractors = random.sample(distractor_pool, total_docs - 1)
    docs = distractors.copy()
    docs.insert(gold_position, gold_doc)
    return docs


def format_prompt(
    documents: list[str],
    question: str,
    preamble: str = "",
) -> str:
    """Format the multi-document QA prompt."""
    parts = []
    if preamble:
        parts.append(preamble.strip())
        parts.append("")

    n = len(documents)
    parts.append(
        f"Below are {n} documents. Use only the information in these documents "
        f'to answer the question. If the answer is not in the documents, say "I don\'t know."'
    )
    parts.append("")

    for i, doc in enumerate(documents, 1):
        parts.append(f"Document [{i}]: {doc}")

    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer with only the answer, no explanation:")

    return "\n".join(parts)


def format_closedbook_prompt(question: str) -> str:
    """Format a closed-book prompt (no documents)."""
    return (
        f"Answer the following trivia question. If you don't know, say "
        f'"I don\'t know."\n\n'
        f"Question: {question}\n"
        f"Answer with only the answer, no explanation:"
    )
