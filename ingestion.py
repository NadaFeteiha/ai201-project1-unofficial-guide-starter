"""
Milestone 3 — Document ingestion and chunking.

Loads the reader-review .txt files from documents/, splits them into
individual reviews (capturing star rating + review number as metadata),
then chunks each review with LangChain's RecursiveCharacterTextSplitter
using the chunk size / overlap specified in planning.md.

Run directly to inspect the output:
    python ingestion.py
"""

import os
import re
import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config (matches planning.md Chunking Strategy) ---
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

# Matches "REVIEW 3 (5 stars):" and captures the number and the rating.
REVIEW_HEADER = re.compile(r"REVIEW\s+(\d+)\s*\((\d+)\s*stars?\)\s*:", re.IGNORECASE)


def load_documents(documents_dir=DOCUMENTS_DIR):
    """Read every .txt file in documents_dir.

    Returns a list of dicts: {filename, book_title, raw_text}.
    """
    docs = []
    paths = sorted(glob.glob(os.path.join(documents_dir, "*.txt")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        filename = os.path.basename(path)
        # First line is the title header, e.g. "And Then There Were None - Reader Reviews"
        first_line = raw.splitlines()[0] if raw.strip() else ""
        book_title = first_line.replace("- Reader Reviews", "").strip()
        docs.append({"filename": filename, "book_title": book_title, "raw_text": raw})
    return docs


def parse_reviews(doc):
    """Split one document's raw text into individual reviews.

    Returns a list of dicts: {review_number, rating, text}.
    """
    raw = doc["raw_text"]
    reviews = []

    # Find every review header and slice the text between consecutive headers.
    matches = list(REVIEW_HEADER.finditer(raw))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = clean_text(raw[start:end])
        if text:
            reviews.append(
                {
                    "review_number": int(m.group(1)),
                    "rating": int(m.group(2)),
                    "text": text,
                }
            )
    return reviews


def clean_text(text):
    """Light cleaning: collapse whitespace, strip stray separators/quotes padding.

    The source files are already plain text (no HTML), so this mostly
    normalizes blank lines and surrounding whitespace.
    """
    text = text.replace("“", '"').replace("”", '"')  # smart quotes
    text = text.replace("—", "-")  # em dash
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces/tabs
    text = re.sub(r"\n\s*\n+", "\n", text)     # collapse blank lines
    return text.strip()


def build_chunks(documents_dir=DOCUMENTS_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Full pipeline: load -> parse reviews -> chunk, with metadata attached.

    Returns a list of dicts:
        {id, text, source, book_title, review_number, rating, chunk_index}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting on sentence/paragraph boundaries before raw chars.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in load_documents(documents_dir):
        for review in parse_reviews(doc):
            pieces = splitter.split_text(review["text"])
            for j, piece in enumerate(pieces):
                piece = piece.strip()
                if not piece:
                    continue
                chunk_id = f"{doc['filename']}::r{review['review_number']}::c{j}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": piece,
                        "source": doc["filename"],
                        "book_title": doc["book_title"],
                        "review_number": review["review_number"],
                        "rating": review["rating"],
                        "chunk_index": j,
                    }
                )
    return chunks


def _inspect():
    """Print summary stats and a few sample chunks for manual inspection."""
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from '{DOCUMENTS_DIR}/'")
    for d in docs:
        n_reviews = len(parse_reviews(d))
        print(f"  - {d['filename']:<35} {d['book_title']:<35} ({n_reviews} reviews)")

    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    lengths = [len(c["text"]) for c in chunks]
    if lengths:
        print(f"Chunk length — min: {min(lengths)}  max: {max(lengths)}  "
              f"avg: {sum(lengths) // len(lengths)} chars")

    print("\n===== 5 SAMPLE CHUNKS =====")
    # Spread the samples across the corpus rather than all from one file.
    step = max(1, len(chunks) // 5)
    for c in chunks[::step][:5]:
        print(f"\n[{c['id']}]  source={c['source']}  rating={c['rating']}★  "
              f"len={len(c['text'])}")
        print(f"  {c['text']}")


if __name__ == "__main__":
    _inspect()
