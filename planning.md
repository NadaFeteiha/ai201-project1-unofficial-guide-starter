# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

My Unofficial Guide covers **mystery, thriller, and horror book reviews** — specifically, what real readers say about popular books in these genres.

This knowledge is valuable because official sources like publisher descriptions or critic reviews are polished and biased toward selling books. What readers actually experience — the pacing problems, the predictable twists, the disappointing endings, the unexpected emotional impact — lives in informal spaces like Goodreads, Reddit, and Amazon reviews. A reader trying to decide whether a book is right for them cannot easily search across all these opinions and get a grounded, summarized answer. This system makes that kind of informal reader knowledge searchable and answerable.

---

## Documents

8 source files, each containing 8–9 real reader reviews collected from Goodreads. Each file mixes 5-star, 3-star, and 1-star reviews to ensure variety of opinion. Total: approximately 71 real reader reviews.

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Then She Was Gone — Lisa Jewell (Mystery/Thriller) | 8–9 Goodreads reviews, mixed ratings | `documents/then_she_was_gone.txt` (Goodreads) |
| 2 | And Then There Were None — Agatha Christie (Mystery) | 8–9 Goodreads reviews, mixed ratings | `documents/and_then_there_were_none.txt` (Goodreads) |
| 3 | Murder on the Orient Express — Agatha Christie (Mystery) | 8–9 Goodreads reviews, mixed ratings | `documents/murder_on_the_orient_express.txt` (Goodreads) |
| 4 | Death on the Nile — Agatha Christie (Mystery) | 8–9 Goodreads reviews, mixed ratings | `documents/death_on_the_nile.txt` (Goodreads) |
| 5 | Pretend You Don't See Her — Mary Higgins Clark (Mystery/Thriller) | 8–9 Goodreads reviews, mixed ratings | `documents/pretend_you_dont_see_her.txt` (Goodreads) |
| 6 | It — Stephen King (Horror) | 8–9 Goodreads reviews, mixed ratings | `documents/it.txt` (Goodreads) |
| 7 | The Woman in the Window — A.J. Finn (Psychological Thriller) | 8–9 Goodreads reviews, mixed ratings | `documents/woman_in_the_window.txt` (Goodreads) |
| 8 | The Girl on the Train — Paula Hawkins (Psychological Thriller) | 8–9 Goodreads reviews, mixed ratings | `documents/girl_on_the_train.txt` (Goodreads) |

---

## Chunking Strategy

**Chunk size:** 200–300 characters (target `chunk_size=250`)

**Overlap:** 50 characters

**Reasoning:** My documents are short, opinion-based reader reviews — not long-form guides or articles. Each review expresses a distinct opinion in a few sentences. A chunk of 200–300 characters captures roughly one complete thought or opinion (e.g. "The pacing was slow but the ending was worth it").

Smaller chunks work better here because:
- Each chunk will embed a single clear idea, making semantic search more precise.
- A query like "what do readers say about pacing?" should match a chunk specifically about pacing, not a 1000-character chunk that also covers characters, writing style, and plot.

The 50-character overlap ensures that if a key opinion spans two adjacent chunks, at least part of it appears in both — reducing the chance of losing important context at chunk boundaries.

**Risk:** Chunks may sometimes be too small to carry full meaning on their own. For example, "The ending was terrible" without the surrounding context of which book or why. This will be monitored during the chunk inspection step.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`
- Runs locally — no API key or rate limits required
- Good performance on short English text
- Fast and lightweight for a student project

**Top-k:** 5 chunks per query

Retrieving 5 chunks gives the LLM enough context to synthesize an answer from multiple reviewer opinions without diluting the response with loosely related content. Too few (e.g. 2) risks missing the most relevant opinion; too many (e.g. 10) risks pulling in off-topic chunks from other books.

**Production tradeoff reflection (if cost were not a constraint):**
- **OpenAI `text-embedding-3-large`** would offer higher accuracy on nuanced opinion text.
- **Multilingual embeddings** (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) would be worth considering if expanding to non-English reviews.
- **Context length:** `all-MiniLM-L6-v2` has a 256-token limit — fine for short reviews but would need replacement for longer documents.
- **API vs local:** Local embeddings (sentence-transformers) are free and private; API embeddings (OpenAI) are faster and more accurate but cost money per token.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Is And Then There Were None good for first-time mystery readers? | Yes — multiple reviewers recommend it as a starting point for the genre and for Agatha Christie specifically. |
| 2 | What are common complaints about Agatha Christie's writing style? | Reviewers frequently mention sterile/bland writing, carbon-copy characters that are hard to distinguish, lack of descriptive narration, and racist/sexist attitudes that have not aged well. |
| 3 | What do readers say about the pacing of It by Stephen King? | Many readers find the book too long (1000+ pages) and say it bloats beyond what the story needs, though fans argue the length is necessary for character development. |
| 4 | Which book do readers compare most to Gone Girl? | The Girl on the Train is most frequently compared to Gone Girl — reviewers often say it was marketed as "the next Gone Girl" and evaluate whether it lives up to that comparison. |
| 5 | What do readers say about the ending of Gone Girl? | Out-of-scope: Gone Girl is not in our document set. The system should decline to answer ("I don't have enough information") rather than hallucinate. |

---

## Anticipated Challenges

1. **System makes up answers not in the documents (hallucination).** The LLM may draw on its general training knowledge about books rather than the retrieved chunks. For example, if asked about Gone Girl (not in our documents), it might generate a plausible-sounding answer from general knowledge instead of saying it doesn't have enough information. Mitigation: a strict grounding prompt that explicitly instructs the model to answer only from retrieved context and say "I don't have enough information" when the documents don't cover the topic.

2. **Retrieval returns reviews from the wrong book.** Since many reviews mention multiple books for comparison (e.g. "better than Gone Girl" or "similar to Orient Express"), a query about one book might retrieve chunks from a review about a different book. Mitigation: monitor during retrieval testing and attach source-filename metadata to each chunk so answers can be attributed and filtered.

---

## Architecture

```
Documents (8 .txt files)
        |
        v
[ Document Ingestion ]
  - Load raw .txt files
  - Clean text (remove links, separators)
  - Library: Python built-in file I/O
        |
        v
[ Chunking ]
  - Split into 200-300 character chunks
  - 50 character overlap
  - Library: LangChain RecursiveCharacterTextSplitter
        |
        v
[ Embedding + Vector Store ]
  - Embed chunks using all-MiniLM-L6-v2
  - Store in ChromaDB with source metadata
  - Library: sentence-transformers, ChromaDB
        |
        v
[ Retrieval ]
  - Accept user query
  - Return top-5 most relevant chunks + sources
  - Method: semantic similarity search
        |
        v
[ Generation ]
  - Pass retrieved chunks as context to LLM
  - Generate grounded answer with source attribution
  - LLM: Groq llama-3.3-70b-versatile
        |
        v
[ Query Interface ]
  - Gradio web UI
  - Input: user question
  - Output: grounded answer + source documents
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I'll give Claude the **Documents** and **Chunking Strategy** sections plus the pipeline diagram, and ask it to produce (a) a Python script that loads all 8 `.txt` files, cleans them (strip links/separators) using built-in file I/O, and (b) a `chunk_text()` function using LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=250` and `chunk_overlap=50`. I'll verify by inspecting a sample of chunks to confirm each holds roughly one opinion and the sizes/overlap match the spec.

**Milestone 4 — Embedding and retrieval:**
I'll give Claude the **Retrieval Approach** section and the diagram, and ask it to produce (a) a script that embeds all chunks with `all-MiniLM-L6-v2` and stores them in ChromaDB with source-filename metadata, and (b) a `retrieve(query, k=5)` function that returns the top-k chunks and their source filenames. I'll verify by running the 5 evaluation queries and checking the returned chunks come from the correct book.

**Milestone 5 — Generation and interface:**
I'll give Claude the **Evaluation Plan** and the grounding requirement, and ask it to produce (a) a grounding prompt template and a generation function using Groq `llama-3.3-70b-versatile` that answers only from retrieved context and cites sources, and (b) a working `app.py` using Gradio Blocks with a question input, an answer output, and a sources output. I'll verify against all 5 test questions — especially that question 5 (Gone Girl) is correctly declined rather than hallucinated.
