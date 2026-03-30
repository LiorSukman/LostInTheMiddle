"""Stage 5 — Build distractor sets for each verified Q&A pair."""

import json

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import DATASET_DIR, CHUNKED_PASSAGES_DIR

# Domain groupings for finding related articles
DOMAIN_GROUPS = {
    "film_awards": [
        "97th Academy Awards", "98th Academy Awards",
        "82nd Golden Globes", "81st Golden Globes",
        "77th British Academy Film Awards", "78th British Academy Film Awards",
        "2024 in film", "2025 in film",
    ],
    "music": [
        "66th Annual Grammy Awards", "67th Annual Grammy Awards",
        "2024 in music",
    ],
    "sports": [
        "2024 Summer Olympics", "UEFA Euro 2024",
        "2024 FIFA Intercontinental Cup", "2025 Australian Open",
        "2024 Wimbledon Championships", "2024 Tour de France",
    ],
    "science": [
        "Nobel Prize in Physics", "Nobel Prize in Physiology or Medicine",
        "2024 Nobel Prizes",
    ],
    "politics": [
        "2024 United States presidential election", "2025 in politics",
    ],
    "literature": [
        "2024 Booker Prize", "2024 Pulitzer Prize", "2025 in literature",
    ],
}


def get_domain(article_title: str) -> str:
    """Find which domain group an article belongs to."""
    for domain, articles in DOMAIN_GROUPS.items():
        if article_title in articles:
            return domain
    return "other"


def get_related_articles(article_title: str) -> list[str]:
    """Get articles in the same domain group."""
    domain = get_domain(article_title)
    return DOMAIN_GROUPS.get(domain, [])


def answer_in_text(answer: str, text: str) -> bool:
    """Check if the answer string appears in the text (case-insensitive)."""
    return answer.lower() in text.lower()


def normalize_article_name(name: str) -> str:
    """Normalize article name for fuzzy matching."""
    return name.lower().strip()


def articles_match(name1: str, name2: str) -> bool:
    """Check if two article names refer to the same article."""
    n1 = normalize_article_name(name1)
    n2 = normalize_article_name(name2)
    if n1 == n2:
        return True
    # Check if one is a substring of the other
    if n1 in n2 or n2 in n1:
        return True
    # Stem-aware word overlap: strip trailing 's' for plural matching
    def stems(words):
        return {w.rstrip('s') for w in words}
    words1 = set(n1.split())
    words2 = set(n2.split())
    stem_overlap = len(stems(words1) & stems(words2))
    overlap = stem_overlap / max(min(len(words1), len(words2)), 1)
    return overlap >= 0.7


def build_gold_from_raw_article(qa: dict) -> str | None:
    """Fallback: build gold passage directly from the raw article text."""
    from .config import RAW_ARTICLES_DIR
    import re

    answer = qa["answer"]
    source = qa["source_sentence"]
    article = qa["article"]

    # Find the raw article file
    meta_path = RAW_ARTICLES_DIR / "metadata.json"
    if not meta_path.exists():
        return None

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    article_text = None
    for m in metadata:
        if articles_match(m["title"], article):
            filepath = RAW_ARTICLES_DIR / m["filename"]
            if filepath.exists():
                article_text = filepath.read_text(encoding="utf-8")
                break

    if article_text is None:
        return None

    # Find the answer in the article text
    answer_lower = answer.lower()
    text_lower = article_text.lower()
    pos = text_lower.find(answer_lower)
    if pos == -1:
        return None

    # Extract a passage around the answer (targeting 60-100 words)
    # Find sentence boundaries around the answer
    sentences = re.split(r'(?<=[.!?])\s+', article_text)
    target_sentences = []
    found = False
    for sent in sentences:
        if answer_lower in sent.lower():
            target_sentences.append(sent)
            found = True
        elif found and len(' '.join(target_sentences).split()) < 60:
            target_sentences.append(sent)
        elif not found and target_sentences:
            # Previous context sentence
            pass

    if not target_sentences:
        return None

    passage = ' '.join(target_sentences)
    # Ensure answer is still present
    if answer_lower not in passage.lower():
        return None

    # Add context if too short
    if len(passage.split()) < 40:
        # Try to find the sentence index and add neighbors
        for i, sent in enumerate(sentences):
            if answer_lower in sent.lower():
                start = max(0, i - 1)
                end = min(len(sentences), i + 3)
                passage = ' '.join(sentences[start:end])
                break

    return passage


def build_gold_passage(qa: dict, all_chunks: list[dict]) -> str | None:
    """Build an expanded gold passage from the source sentence.

    Find the chunk containing the source sentence and return it (already
    60-150 words typically). Verify the answer is present.
    """
    source = qa["source_sentence"]
    answer = qa["answer"]
    article = qa["article"]

    # Find chunks from the same article that contain the source sentence
    best_chunk = None
    best_overlap = 0
    for chunk in all_chunks:
        if not articles_match(chunk["article"], article):
            continue
        # Check if source sentence (or a substantial part) is in this chunk
        # Use a fuzzy approach: check if most words from source appear in chunk
        source_words = set(source.lower().split())
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(source_words & chunk_words) / max(len(source_words), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    if best_chunk is not None and best_overlap >= 0.4:
        passage = best_chunk["text"]
        if answer_in_text(answer, passage):
            return passage

    # Fallback: find any chunk from the same article that contains the answer
    # Pick the one with highest source sentence overlap
    fallback_chunk = None
    fallback_overlap = 0
    for chunk in all_chunks:
        if not articles_match(chunk["article"], article):
            continue
        if not answer_in_text(answer, chunk["text"]):
            continue
        source_words = set(source.lower().split())
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(source_words & chunk_words) / max(len(source_words), 1)
        if overlap > fallback_overlap:
            fallback_overlap = overlap
            fallback_chunk = chunk

    if fallback_chunk is not None:
        return fallback_chunk["text"]

    # Last resort: build passage directly from raw article text
    return build_gold_from_raw_article(qa)


def run():
    """Build distractor sets for all verified Q&A pairs."""
    print("=" * 60)
    print("STAGE 5 — Building distractor sets")
    print("=" * 60)

    # Load inputs
    qa_path = DATASET_DIR / "verified_qa_pairs.json"
    if not qa_path.exists():
        raise FileNotFoundError("Run Stage 4 first — verified_qa_pairs.json not found.")

    verified_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    chunks_path = CHUNKED_PASSAGES_DIR / "all_chunks.json"
    all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    print(f"Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Pre-compute all chunk embeddings
    print(f"Encoding {len(all_chunks)} chunks...")
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=64)

    dataset_items = []
    skipped = 0

    for qa in tqdm(verified_pairs, desc="Building distractor sets"):
        answer = qa["answer"]
        article = qa["article"]
        question = qa["question"]

        # Build gold passage
        gold_passage = build_gold_passage(qa, all_chunks)
        if gold_passage is None:
            print(f"  WARNING: Could not build gold passage for: {question[:60]}...")
            skipped += 1
            continue

        # Find the gold chunk ID
        gold_chunk_id = None
        for chunk in all_chunks:
            if chunk["text"] == gold_passage:
                gold_chunk_id = chunk["chunk_id"]
                break

        # Collect candidate distractors
        related_articles = get_related_articles(article)
        candidates = []
        candidate_indices = []

        for i, chunk in enumerate(all_chunks):
            # Skip the gold chunk itself
            if chunk["text"] == gold_passage:
                continue
            # Skip chunks containing the answer
            if answer_in_text(answer, chunk["text"]):
                continue
            # Prefer same article + related articles, but include all as fallback
            candidates.append(chunk)
            candidate_indices.append(i)

        if len(candidates) < 19:
            print(f"  WARNING: Only {len(candidates)} distractor candidates for: {question[:60]}...")
            skipped += 1
            continue

        # Rank by similarity to question
        q_emb = model.encode([question])
        candidate_embs = chunk_embeddings[candidate_indices]
        scores = np.dot(candidate_embs, q_emb.T).flatten()

        # Prioritize same-article and same-domain chunks with a small boost
        for j, chunk in enumerate(candidates):
            if articles_match(chunk["article"], article):
                scores[j] += 0.05  # small boost for same-article
            elif any(articles_match(chunk["article"], ra) for ra in related_articles):
                scores[j] += 0.02  # smaller boost for related articles

        ranked_indices = np.argsort(scores)[::-1][:19]

        distractors = []
        for rank, idx in enumerate(ranked_indices):
            chunk = candidates[idx]
            distractors.append({
                "rank": rank + 1,
                "text": chunk["text"],
                "article": chunk["article"],
                "chunk_id": chunk["chunk_id"],
                "similarity_score": round(float(scores[idx]), 4),
            })

        item_id = f"item_{len(dataset_items):04d}"
        dataset_items.append({
            "id": item_id,
            "question": question,
            "answer": answer,
            "answer_aliases": [answer],
            "gold_passage": gold_passage,
            "gold_article": article,
            "gold_chunk_id": gold_chunk_id,
            "distractors": distractors,
        })

    # Save final dataset
    out_path = DATASET_DIR / "final_dataset.json"
    out_path.write_text(
        json.dumps(dataset_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone: {len(dataset_items)} dataset items built, {skipped} skipped.")
    print(f"Saved to {out_path}")
    return dataset_items


if __name__ == "__main__":
    run()
