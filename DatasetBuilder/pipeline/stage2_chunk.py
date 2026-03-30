"""Stage 2 — Chunk harvested articles into passages."""

import json
import re

from tqdm import tqdm

from .config import RAW_ARTICLES_DIR, CHUNKED_PASSAGES_DIR


# Chunking parameters
MIN_WORDS = 30
TARGET_MIN_WORDS = 80
TARGET_MAX_WORDS = 115
HARD_MAX_WORDS = 200  # split mid-paragraph above this


def sanitize_title(title: str) -> str:
    return re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()


def is_list_chunk(text: str) -> bool:
    """Detect chunks that are primarily lists of names/numbers with no prose."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    # If most lines are very short (likely list items) and few have verbs
    short_lines = sum(1 for l in lines if len(l.split()) <= 5)
    return short_lines / len(lines) > 0.7


def parse_sections(article_text: str) -> list[tuple[str, str]]:
    """Parse article into (section_header, paragraph_text) tuples."""
    sections = []
    current_section = "Introduction"

    # Split on section headers (== Header ==)
    lines = article_text.split('\n')
    current_paragraphs = []

    for line in lines:
        # Match Wikipedia section headers: == Title == or === Title ===
        header_match = re.match(r'^={2,}\s*(.+?)\s*={2,}$', line.strip())
        if header_match:
            # Flush current paragraphs under previous section
            if current_paragraphs:
                text = '\n'.join(current_paragraphs).strip()
                if text:
                    sections.append((current_section, text))
            current_section = header_match.group(1).strip()
            current_paragraphs = []
        else:
            current_paragraphs.append(line)

    # Flush last section
    if current_paragraphs:
        text = '\n'.join(current_paragraphs).strip()
        if text:
            sections.append((current_section, text))

    return sections


def split_paragraph(text: str, target_max: int = TARGET_MAX_WORDS) -> list[str]:
    """Split a long paragraph into smaller chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_wc = 0

    for sentence in sentences:
        s_wc = len(sentence.split())
        if current_wc + s_wc > target_max and current:
            chunks.append(' '.join(current))
            current = [sentence]
            current_wc = s_wc
        else:
            current.append(sentence)
            current_wc += s_wc

    if current:
        chunks.append(' '.join(current))
    return chunks


def chunk_article(title: str, text: str) -> list[dict]:
    """Chunk a single article into passage dicts."""
    sections = parse_sections(text)
    chunks = []
    chunk_idx = 0
    title_slug = sanitize_title(title)

    for section_name, section_text in sections:
        # Skip sections that are typically not useful
        if section_name.lower() in ("see also", "references", "external links",
                                     "further reading", "notes", "bibliography"):
            continue

        # Split section into paragraphs (double newline)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', section_text) if p.strip()]

        for para in paragraphs:
            word_count = len(para.split())

            if word_count > HARD_MAX_WORDS:
                # Split long paragraphs
                sub_chunks = split_paragraph(para)
            else:
                sub_chunks = [para]

            for chunk_text in sub_chunks:
                wc = len(chunk_text.split())
                if wc < MIN_WORDS:
                    continue
                if is_list_chunk(chunk_text):
                    continue

                # Prepend section header
                prefixed_text = f"[Section: {section_name}] {chunk_text}"

                chunks.append({
                    "article": title,
                    "section": section_name,
                    "chunk_id": f"{title_slug}_{chunk_idx:03d}",
                    "text": prefixed_text,
                    "word_count": wc,
                })
                chunk_idx += 1

    return chunks


def run():
    """Chunk all harvested articles."""
    print("=" * 60)
    print("STAGE 2 — Chunking articles into passages")
    print("=" * 60)

    meta_path = RAW_ARTICLES_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run Stage 1 first — metadata.json not found.")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    all_chunks = []

    for article_meta in tqdm(metadata, desc="Chunking articles"):
        title = article_meta["title"]
        filepath = RAW_ARTICLES_DIR / article_meta["filename"]
        text = filepath.read_text(encoding="utf-8")

        chunks = chunk_article(title, text)

        # Save per-article chunks
        out_name = sanitize_title(title) + ".json"
        out_path = CHUNKED_PASSAGES_DIR / out_name
        out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        all_chunks.extend(chunks)

    # Save flat index of all chunks
    all_path = CHUNKED_PASSAGES_DIR / "all_chunks.json"
    all_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone: {len(all_chunks)} total chunks from {len(metadata)} articles.")
    print(f"Saved to {CHUNKED_PASSAGES_DIR}")

    # Summary stats
    wcs = [c["word_count"] for c in all_chunks]
    print(f"Word count range: {min(wcs)}–{max(wcs)}, median: {sorted(wcs)[len(wcs)//2]}")
    return all_chunks


if __name__ == "__main__":
    run()
