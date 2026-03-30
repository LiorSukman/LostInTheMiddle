"""Build a 30-document variant of the dataset (29 distractors per question).

Reads the existing final_dataset.json (20-doc version) for question/answer/gold
data, then re-ranks the full chunk pool to select the top 29 distractors per
question using the same similarity-based approach as stage5.

Outputs: final_dataset_30.json
The original final_dataset.json is NOT modified.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATASET_DIR = Path(__file__).parent / "dataset"
CHUNKED_PASSAGES_DIR = DATASET_DIR / "chunked_passages"

NUM_DISTRACTORS = 29

# Reuse domain logic from stage5
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.stage5_distractors import (
    get_related_articles,
    answer_in_text,
    articles_match,
)


def main():
    # Load existing dataset (for question/answer/gold passage metadata)
    src_path = DATASET_DIR / "final_dataset.json"
    items = json.loads(src_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(items)} items from final_dataset.json")

    # Load chunk pool
    chunks_path = CHUNKED_PASSAGES_DIR / "all_chunks.json"
    all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Chunk pool: {len(all_chunks)} chunks")

    # Encode all chunks
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Encoding {len(all_chunks)} chunks...")
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=64)

    output_items = []
    skipped = 0

    for item in tqdm(items, desc="Building 30-doc distractors"):
        answer = item["answer"]
        article = item["gold_article"]
        question = item["question"]
        gold_passage = item["gold_passage"]

        related_articles = get_related_articles(article)

        # Collect candidates (exclude gold passage and answer-containing chunks)
        candidates = []
        candidate_indices = []
        for i, chunk in enumerate(all_chunks):
            if chunk["text"] == gold_passage:
                continue
            if answer_in_text(answer, chunk["text"]):
                continue
            candidates.append(chunk)
            candidate_indices.append(i)

        if len(candidates) < NUM_DISTRACTORS:
            print(f"  WARNING: Only {len(candidates)} candidates for: {question[:60]}...")
            skipped += 1
            continue

        # Rank by similarity to question
        q_emb = model.encode([question])
        candidate_embs = chunk_embeddings[candidate_indices]
        scores = np.dot(candidate_embs, q_emb.T).flatten()

        # Domain boosts (same as stage5)
        for j, chunk in enumerate(candidates):
            if articles_match(chunk["article"], article):
                scores[j] += 0.05
            elif any(articles_match(chunk["article"], ra) for ra in related_articles):
                scores[j] += 0.02

        ranked_indices = np.argsort(scores)[::-1][:NUM_DISTRACTORS]

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

        output_items.append({
            "id": item["id"],
            "question": question,
            "answer": answer,
            "answer_aliases": item["answer_aliases"],
            "gold_passage": gold_passage,
            "gold_article": article,
            "gold_chunk_id": item["gold_chunk_id"],
            "distractors": distractors,
        })

    out_path = DATASET_DIR / "final_dataset_30.json"
    out_path.write_text(
        json.dumps(output_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone: {len(output_items)} items with {NUM_DISTRACTORS} distractors each.")
    print(f"Skipped: {skipped}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
