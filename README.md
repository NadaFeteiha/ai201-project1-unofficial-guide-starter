# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that lets you search what real readers say
about mystery, thriller, and horror books. You ask a question in plain language and get an
answer that is based on real Goodreads reviews, with the sources listed.

---

## Domain

This system is about **reader reviews of mystery, thriller, and horror books** — what real
readers actually say about popular books in these genres.

This is useful because official sources (like the publisher's description or professional
critics) are polished and try to sell the book. What readers really feel — slow pacing,
predictable twists, weak endings — is spread across places like Goodreads, Reddit, and
Amazon. A reader who wants to decide if a book is right for them cannot easily search all
those scattered opinions and get one clear answer. This system does that for them.

---

## Document Sources

8 plain-text files. Each file has 8–9 real reader reviews from Goodreads. Each file mixes
5-star, 3-star, and 1-star reviews on purpose, so we get different opinions.
**Total: 69 reviews → 359 chunks.**

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | And Then There Were None — Agatha Christie (Mystery) | Goodreads reviews (.txt) | `documents/and_then_there_were_none.txt` |
| 2 | Murder on the Orient Express — Agatha Christie (Mystery) | Goodreads reviews (.txt) | `documents/murder_on_the_orient_express.txt` |
| 3 | Death on the Nile — Agatha Christie (Mystery) | Goodreads reviews (.txt) | `documents/death_on_the_nile.txt` |
| 4 | It — Stephen King (Horror) | Goodreads reviews (.txt) | `documents/it.txt` |
| 5 | The Woman in the Window — A.J. Finn (Psych. Thriller) | Goodreads reviews (.txt) | `documents/woman_in_the_window.txt` |
| 6 | The Girl on the Train — Paula Hawkins (Psych. Thriller) | Goodreads reviews (.txt) | `documents/girl_on_the_train.txt` |
| 7 | Then She Was Gone — Lisa Jewell (Mystery/Thriller) | Goodreads reviews (.txt) | `documents/then_she_was_gone.txt` |
| 8 | Pretend You Don't See Her — Mary Higgins Clark (Mystery/Thriller) | Goodreads reviews (.txt) | `documents/pretend_you_dont_see_her.txt` |

---

## Chunking Strategy

**Chunk size:** 250 characters (range 200–300)

**Overlap:** 50 characters

**Why this fits the documents:** The documents are short reviews, not long articles. Each
review says a few short things. A chunk of about 250 characters holds about one full
opinion (for example, "the pacing was slow but the ending was worth it"). So each chunk has
one clear idea, and search can match a question about *pacing* to a chunk about *pacing*,
instead of a big chunk that mixes pacing, characters, and plot together. The 50-character
overlap means if one opinion sits on the line between two chunks, part of it stays in both
chunks, so we do not lose it.

**Preprocessing before chunking** (`ingestion.py`): each file is loaded, the title line is
removed, and the text is split into single reviews using the `REVIEW N (M stars):` markers.
We save the **review number** and **star rating** as metadata. We also clean up extra spaces
and smart quotes. Then each review is split with LangChain's `RecursiveCharacterTextSplitter`
(`chunk_size=250`, `chunk_overlap=50`). Every chunk keeps its source file name, book title,
review number, and rating.

**Final chunk count:** 359 chunks from 8 documents (69 reviews). Average chunk is about 169
characters.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` from `sentence-transformers`. It runs on my computer with
no API key and no rate limits, it works well on short English text, and it is fast and small
— a good fit for short reviews. The chunks are stored in a local **ChromaDB** database that
uses **cosine distance**, and we retrieve the **top 5** chunks per question.

**If I deployed this for real users (cost not a problem), I would think about:**
- **Accuracy on opinions:** A bigger model like OpenAI `text-embedding-3-large` would
  understand subtle feelings better (for example, "loved the twist" vs. "saw the twist
  coming"), so retrieval would be more correct.
- **Context length:** `all-MiniLM-L6-v2` only reads up to 256 tokens. That is fine for short
  reviews, but it would cut off long articles. For long text I would need a model with a
  longer limit.
- **Other languages:** For non-English reviews I would use a multilingual model like
  `paraphrase-multilingual-MiniLM-L12-v2`.
- **Local vs. API:** A local model is free and private but slower for big datasets. An API
  model is faster and more accurate at scale, but it costs money per token and needs internet.

---

## Grounded Generation

The LLM is **Groq `llama-3.3-70b-versatile`**. We make sure the answer stays grounded
(based only on the reviews) in two ways:

**1. The system prompt** (`generate.py`) tells the model to use only the given reviews:

> Answer using only information found in the provided reviews. Do not use any outside
> knowledge about these books, even if you know it. If the reviews do not contain enough
> information to answer the question, reply exactly: "I don't have enough information on
> that." Do not guess or fill in from general knowledge.

We also use a low temperature (0.2) so the model stays close to the text. The retrieved
chunks are put into a numbered list, and each one shows its source file and rating.

**2. Source attribution is added by the code, not the model.** After the model answers, the
code adds a `Sources:` line built from the retrieved chunks' metadata. So the sources are
always correct and do not depend on the model remembering them. If the model says it does
not have enough information, we do not add any sources.

---

## Sample Chunks

Five example chunks, each with its source file:

1. **`and_then_there_were_none.txt`** (5★) — *"I'm a big lover of Agatha Christie, she has
   written some fantastic murder mysteries and her stories never get tiring. But this is the
   one that just comes out on top every time"*
2. **`death_on_the_nile.txt`** (1★) — *"This DOES NOT stand the test of time! This was my
   second Hercule Poirot novel after Murder on the Orient Express and my opinion remains
   unchanged. I genuinely cannot fathom how Agatha Christie's books are rated so highly on
   goodreads"*
3. **`it.txt`** (3★) — *"With this one King threw a kitchen sink full of monsters into this
   with the villain able to take the form of whatever will scare its latest victim the most"*
4. **`pretend_you_dont_see_her.txt`** (5★) — *"Okay, so Pretend you Don't See Her is my first
   Mary Higgins Clark book I've ever read, and after having just finished it, I can guarantee
   it won't be my last"*
5. **`then_she_was_gone.txt`** (2★) — *"I saw the twist coming a mile away because I recall a
   similar plot on an old Law & Order: Special Victims Unit episode nearly a decade ago. And
   the episode was way more believable than what I read."*

---

## Retrieval Test Results

Three questions with their top retrieved chunks (cosine distance — a smaller number means a
closer match; under 0.5 is a strong match):

**Question 1: "Is And Then There Were None a good first mystery book to read?"**
- `and_then_there_were_none.txt` (0.275) — "...absolutely worth adding to your list..."
- `and_then_there_were_none.txt` (0.301) — "...the book that started my absolute love of the mystery genre..."
- `and_then_there_were_none.txt` (0.305) — "...one of Agatha Christie's most celebrated mystery novels..."

*Why these are relevant:* All three top chunks come from the correct book and talk about it
as a good place to start the genre — exactly what the question asks. The distances are well
under 0.5, so the matches are strong.

**Question 2: "Which book do readers compare most to Gone Girl?"**
- `girl_on_the_train.txt` (0.346) — "This has frequently been called the next Gone Girl..."
- `pretend_you_dont_see_her.txt` (0.386) — "...love affair with the suspense genre..."

*Why these are relevant:* The top chunk is the exact review that says *The Girl on the Train*
was called "the next Gone Girl" — a correct, direct match for the question.

**Question 3: "What do readers say about the pacing of It by Stephen King?"**
- `girl_on_the_train.txt` (0.362) — "...The pace was slow and sometimes the chapters were just plain boring..."
- `it.txt` (0.363) — "...the length of the novel. That's a lot of words on a lot of pages..."
- `it.txt` (0.404) — "...pages long? Does Stephen King not have an editor?"

*Note:* The top chunk is a strong match for "slow pace" but it is from the **wrong book**
(*The Girl on the Train*). See the Failure Case Analysis below.

---

## Example Responses

**Example 1 — grounded answer with sources** (Question: "What do readers complain about in
Agatha Christie's writing?"):
> Readers complain that Agatha Christie's stories can seem "too contrived and stiff" with
> some elements being "amateurishly done" (review 2), and that her books "do not stand the
> test of time" (review 3).
>
> **Sources:** and_then_there_were_none.txt, death_on_the_nile.txt

**Example 2 — grounded answer with sources** (Question: "Which book do readers compare most
to Gone Girl?"):
> The book "The Girl on the Train" is compared to Gone Girl by a reader.
>
> **Sources:** girl_on_the_train.txt, pretend_you_dont_see_her.txt, then_she_was_gone.txt, death_on_the_nile.txt

**Example 3 — out-of-scope refusal** (Question: "What do readers say about the ending of
Gone Girl?"). Gone Girl is not in our documents:
> I don't have enough information on that.

The system correctly refuses instead of making up an answer, and it does not add any sources.

---

## Query Interface

A **Gradio** web app (`app.py`). Run `python app.py` and open `http://localhost:7860`.

- **Input — "Your question":** a text box for the user's question (you can also press Enter).
  There are clickable example questions too.
- **Slider — "Chunks to retrieve (top-k)":** lets the user choose how many chunks to retrieve
  (1 to 10, default 5).
- **Output — "Answer":** the grounded answer, with the `Sources:` line at the end.
- **Output — "Retrieved from":** a simple list of the source files used.
- **Output — "Retrieved chunks (with distance scores)":** each retrieved chunk with its
  distance, book title, rating, and text — so you can see exactly what the answer is based on.

**Sample interaction:**
```
Your question: What do readers say about the pacing of It by Stephen King?

Answer: According to the reviews, one reader says that "this never lags too much"
(review [2]), so the pacing is generally fine. But another reader complains about the
length ("ELEVEN-FUCKING-HUNDRED pages long", review [4]), so they found it too long.
Sources: girl_on_the_train.txt, it.txt, woman_in_the_window.txt

Retrieved from:
• girl_on_the_train.txt
• it.txt
• woman_in_the_window.txt
```

---

## Evaluation Report

| # | Question | Expected answer | System response (short) | Retrieval quality | Response accuracy |
|---|----------|-----------------|-------------------------|-------------------|-------------------|
| 1 | Is *And Then There Were None* a good first mystery book to read? | Yes — reviewers say it is a good place to start the genre | Says it is a good first read, but notes no review uses the exact words "first-time" | Relevant | Accurate |
| 2 | What do readers complain about in Agatha Christie's writing? | Plain/sterile writing, similar characters, little description, old-fashioned attitudes | Found "contrived/stiff," "amateurish," "does not age well"; missed the similar-characters and old-attitudes points | Partially relevant | Partially accurate |
| 3 | What do readers say about the pacing of *It*? | Many say it is too long (1000+ pages); fans say the length is needed | Mixed: mentions the length, but also used a chunk from the wrong book; answer is unclear | Off-target (top chunk wrong book) | Partially accurate |
| 4 | Which book do readers compare most to *Gone Girl*? | *The Girl on the Train* (called "the next Gone Girl") | "The Girl on the Train" | Relevant | Accurate |
| 5 | What do readers say about the ending of *Gone Girl*? | Out-of-scope — the system should refuse | "I don't have enough information on that." | Off-target (no source has it) | Accurate (correct refusal) |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "What do readers say about the pacing of *It* by Stephen King?"

**What the system returned:** The top chunk came from `girl_on_the_train.txt` ("The pace was
slow and sometimes the chapters were just plain boring"), not from `it.txt`. The answer was a
mix — it quoted "this never lags too much" from one *It* review and the 1,100-page complaint
from another — so it was less clear than the expected answer (most readers find *It* too long,
but fans defend the length).

**Root cause (which part of the pipeline):** This is a **retrieval** problem caused by the
**chunking + embedding** steps. My chunks are small (~250 characters) and each one holds one
idea. A chunk that says "the pace was slow and the chapters were boring" is a very strong
match for a *pacing* question — no matter which book it is about. Because the book title is
often not repeated inside every chunk, the embedding has no way to prefer the correct book.
So a general "slow pacing" comment about *The Girl on the Train* beats the real *It* reviews.
This is exactly the "retrieval returns reviews from the wrong book" risk I wrote about in
`planning.md`.

**How I would fix it:** (1) **Metadata filtering** — find the book named in the question and
only search chunks from that book (a ChromaDB `where` filter on the `source`/`book_title`
metadata I already save). (2) **Add the book title to each chunk before embedding** so the
title is part of the embedding and same-book chunks score higher. (3) **Hybrid search** —
mix semantic search with a keyword (BM25) match on the title to push the right book up.

---

## Spec Reflection

**One way the spec helped me:** Writing the Chunking Strategy and Retrieval Approach parts of
`planning.md` before coding gave the AI tool clear targets (250-character chunks, 50 overlap,
`all-MiniLM-L6-v2`, top-5, ChromaDB with source metadata). So the generated `ingestion.py`
and `retrieval.py` matched my plan from the start instead of using generic defaults. The
Anticipated Challenges part also helped: because I had predicted "retrieval returns reviews
from the wrong book," I quickly recognized the *It* pacing result as a known risk, not a
random bug — and I already had source metadata on every chunk to support a future fix.

**One way my implementation was different from the spec:** The spec assumed each review has a
whole-number star rating, so I planned to store `rating` as an integer. While checking chunks,
I found some reviews are rated `4.5 stars`. My first regex only matched whole numbers, so
those reviews were skipped and their header text leaked into chunks. I changed the code to
read fractional ratings and store `rating` as a float. This was different from the plan, but I
made the change because the real data did not match my first assumption.

---

## AI Usage

**Instance 1 — Ingestion and chunking**
- *What I gave the AI:* The Documents and Chunking Strategy parts of my `planning.md`, the
  pipeline diagram, and the real format of my review files (a title line plus
  `REVIEW N (M stars):` markers).
- *What it produced:* `ingestion.py` — it loads all 8 files, splits them into single reviews
  with rating and number metadata, and chunks them with `RecursiveCharacterTextSplitter` at
  `chunk_size=250`, `chunk_overlap=50`.
- *What I changed:* During the chunk-inspection step I found a `REVIEW 4 (4.5 stars):` header
  leaking into a chunk because the regex only matched whole-number ratings. I directed the fix
  to handle fractional ratings and store the rating as a float. This recovered a review that
  had been dropped (Then She Was Gone went from 7 to 8 reviews).

**Instance 2 — Embedding, retrieval, and grounded generation**
- *What I gave the AI:* My Retrieval Approach section and the rule that answers must be
  grounded and must cite sources.
- *What it produced:* `retrieval.py` (embed with `all-MiniLM-L6-v2`, store in ChromaDB,
  `retrieve(query, k=5)`) and `generate.py` (the grounding prompt and the Groq call).
- *What I changed:* I set the vector store to use **cosine distance** (not the default L2) so
  the scores are easy to read (0–2) and match the "under 0.5 is good" rule I use. I also made
  source attribution happen in **code** from the metadata, instead of trusting the model to
  cite sources, and I turned off citations when the answer is an out-of-scope refusal.
