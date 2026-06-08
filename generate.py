"""
Milestone 5 — Grounded response generation.

ask(question) retrieves the top-k chunks, passes them to Groq's
llama-3.3-70b-versatile with a strict grounding prompt, and returns
the answer plus the list of source documents it drew from.

Grounding is enforced two ways:
  1. The system prompt instructs the model to answer ONLY from the
     provided context and to refuse when the context is insufficient.
  2. Source attribution is appended programmatically from retrieval
     metadata, so citations don't depend on the model remembering to add them.

Test from the command line:
    python generate.py "What do readers say about the pacing of It?"
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieval import retrieve, TOP_K

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about books \
using ONLY the reader reviews provided in the context below.

Rules:
- Answer using only information found in the provided reviews. Do not use any outside \
knowledge about these books, even if you know it.
- If the reviews do not contain enough information to answer the question, reply \
exactly: "I don't have enough information on that." Do not guess or fill in from \
general knowledge.
- Base your answer on what readers actually say. You may summarize and synthesize \
across multiple reviews.
- Be concise and specific."""


def _format_context(hits):
    """Render retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] (from {h['source']}, {h['rating']} stars)\n{h['text']}"
        )
    return "\n\n".join(blocks)


def ask(question, k=TOP_K):
    """Answer a question grounded in retrieved reviews.

    Returns {answer, sources, hits}:
      - answer: the model's grounded response (citations appended)
      - sources: de-duplicated list of source filenames used as context
      - hits: the raw retrieved chunks (for evaluation / debugging)
    """
    hits = retrieve(question, k=k)
    context = _format_context(hits)

    user_prompt = (
        f"Context (reader reviews):\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature -> stay close to the source text
    )
    answer = response.choices[0].message.content.strip()

    # De-duplicate sources, preserving retrieval order.
    sources = list(dict.fromkeys(h["source"] for h in hits))

    # Only attach citations when the model actually answered from context.
    if "i don't have enough information" not in answer.lower():
        answer += "\n\nSources: " + ", ".join(sources)

    return {"answer": answer, "sources": sources, "hits": hits}


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What do readers say about the pacing of It?"
    result = ask(q)
    print(f"Q: {q}\n")
    print(result["answer"])
