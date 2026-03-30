import re


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def score(response: str, answer_aliases: list[str]) -> bool:
    """Check if any answer alias appears in the model response.

    TriviaQA provides multiple valid answer forms (e.g. "JFK", "John Kennedy").
    We check containment rather than exact match since models sometimes add
    minor phrasing around the answer.
    """
    norm_response = normalize(response)
    return any(normalize(alias) in norm_response for alias in answer_aliases)
