# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that makes informal reader reviews of
mystery, thriller, and horror books searchable and answerable. Ask a plain-language
question and get a grounded, source-cited answer drawn from real Goodreads reviews.

---

## Domain

This system covers **mystery, thriller, and horror book reviews** — specifically, what
real readers say about popular books in these genres.

This knowledge is valuable because official sources (publisher blurbs, professional
critic reviews) are polished and biased toward selling books. What readers actually
experience — slow pacing, predictable twists, disappointing endings, unexpected
emotional impact — lives in informal spaces like Goodreads, Reddit, and Amazon. A reader
deciding whether a book is right for them can't easily search across all those scattered
opinions and get a grounded, summarized answer. This system makes that informal reader
knowledge searchable.

---

## Document Sources

8 plain-text files, each containing 8–9 real reader reviews collected from Goodreads.
Each file deliberately mixes 5-star, 3-star, and 1-star reviews to capture a range of
opinion. **Total: 69 reviews → 359 chunks.**

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

**Chunk size:** 250 characters (target range 200–300)

**Overlap:** 50 characters

**Why these choices fit your documents:** The documents are short, opinion-based reader
reviews, not long-form guides. Each review expresses a few distinct thoughts in a handful
of sentences. A ~250-character chunk captures roughly one complete opinion (e.g. "the
pacing was slow but the ending was worth it"), so each embedding encodes a single clear
idea and semantic search can match a query about *pacing* to a chunk specifically about
pacing — rather than to a 1,000-character chunk that also covers characters, plot, and
style. The 50-character overlap means that if one opinion spans a chunk boundary, part of
it still appears in both chunks.

**Preprocessing before chunking** (`ingestion.py`): each file is loaded, its title header
line is stripped, and the text is split into individual reviews using a regex on the
`REVIEW N (M stars):` markers — capturing the **review number** and **star rating** as
metadata. Whitespace and smart quotes are normalized. Each review is then chunked with
LangChain's `RecursiveCharacterTextSplitter` (`chunk_size=250`, `chunk_overlap=50`,
splitting preferentially on paragraph → sentence → word boundaries). Every chunk carries
its source filename, book title, review number, and rating.

**Final chunk count:** 359 chunks across 8 documents (69 reviews). Average chunk length
~169 characters.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. It runs locally with no
API key or rate limits, performs well on short English text, and is fast and lightweight
— a good fit for a student project with short review text. Chunks are stored in a local
**ChromaDB** collection configured for **cosine distance**, with `top-k = 5`.

**Production tradeoff reflection:** If I were deploying this for real users and cost
weren't a constraint, I'd weigh:
- **Accuracy on nuanced opinion text:** A larger model like OpenAI `text-embedding-3-large`
  would better distinguish subtle sentiment ("loved the twist" vs. "saw the twist coming"),
  improving retrieval precision on opinion-heavy queries.
- **Context length:** `all-MiniLM-L6-v2` truncates at 256 tokens. Fine for short reviews,
  but it would silently drop content if I expanded to long-form guides or full articles —
  a longer-context model would be necessary there.
- **Multilingual support:** For non-English reviews I'd switch to something like
  `paraphrase-multilingual-MiniLM-L12-v2`.
- **Local vs. API / latency:** Local embeddings are free and private but slower to index
  large corpora; an API model is faster and more accurate at scale but adds per-token cost
  and a network dependency.

---

## Grounded Generation

The LLM is **Groq `llama-3.3-70b-versatile`**. Grounding is enforced two ways:

**System prompt grounding instruction** (`generate.py`): the model is told to answer
using **only** the provided reviews, with explicit rules:

> Answer using only information found in the provided reviews. Do not use any outside
> knowledge about these books, even if you know it. If the reviews do not contain enough
> information to answer the question, reply exactly: "I don't have enough information on
> that." Do not guess or fill in from general knowledge.

Generation runs at a low temperature (0.2) to keep responses close to the source text.
The retrieved chunks are formatted into a numbered context block (each labeled with its
source file and star rating) so the model can reference specific reviews.

**How source attribution is surfaced in the response:** Source filenames are **appended
programmatically** from the retrieval metadata after generation — they are not left to the
model to remember. The de-duplicated list of source files for the retrieved chunks is added
as a `Sources:` line. Attribution is skipped when the model declines to answer (so an
out-of-scope refusal doesn't get spurious citations).

---

## Sample Chunks

Five representative chunks, each with its source document:

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

Three queries with their top retrieved chunks (cosine distance — lower is more similar):

**Query 1: "Is And Then There Were None good for first-time mystery readers?"**
- `and_then_there_were_none.txt` (0.337) — "...one of Agatha Christie's most celebrated mystery novels..."
- `and_then_there_were_none.txt` (0.339) — "...the book that started my absolute love of the mystery genre..."
- `and_then_there_were_none.txt` (0.386) — "...absolutely worth adding to your list..."

*Why relevant:* All three top hits come from the correct book and speak directly to it as
a genre entry point — exactly what the query asks. Distances well under 0.5 indicate strong
matches.

**Query 2: "Which book do readers compare most to Gone Girl?"**
- `girl_on_the_train.txt` (0.346) — "This has frequently been called the next Gone Girl..."
- `pretend_you_dont_see_her.txt` (0.386) — "...love affair with the suspense genre..."

*Why relevant:* The #1 hit is the exact review stating *The Girl on the Train* was marketed
as "the next Gone Girl" — a precise, correct match for the comparison query.

**Query 3: "What do readers say about the pacing of It by Stephen King?"**
- `girl_on_the_train.txt` (0.362) — "...The pace was slow and sometimes the chapters were just plain boring..."
- `it.txt` (0.363) — "...the length of the novel. That's a lot of words on a lot of pages..."
- `it.txt` (0.404) — "...pages long? Does Stephen King not have an editor?"

*Note:* The #1 hit is a strong semantic match for "slow pace" but comes from the **wrong
book** (*The Girl on the Train*). See the Failure Case Analysis below.

---

## Example Responses

**Example 1 — grounded answer with sources** (Q: "What are common complaints about Agatha
Christie's writing style?"):
> One common complaint is that her story elements can seem "too contrived and stiff" and
> "amateurishly done" (review 2). Another complaint is that her books do not "stand the
> test of time" (review 4).
>
> **Sources:** and_then_there_were_none.txt, death_on_the_nile.txt

**Example 2 — grounded answer with sources** (Q: "Which book do readers compare most to
Gone Girl?"):
> The book that a reader compares to Gone Girl is "The Girl on the Train".
>
> **Sources:** girl_on_the_train.txt, pretend_you_dont_see_her.txt, then_she_was_gone.txt, death_on_the_nile.txt

**Example 3 — out-of-scope refusal** (Q: "What do readers say about the ending of Gone
Girl?"). Gone Girl is not in the document set:
> I don't have enough information on that.

The system correctly declines rather than fabricating an answer from the model's general
knowledge, and no citations are appended.

---

## Query Interface

A **Gradio** web UI (`app.py`), launched with `python app.py` at `http://localhost:7860`.

- **Input field — "Your question":** a free-text box for the user's question (also submits
  on Enter). Example prompts are provided as clickable buttons.
- **Output field — "Answer":** the grounded response, including the appended `Sources:` line.
- **Output field — "Retrieved from":** a bulleted list of the source documents the answer
  drew from.

**Sample interaction transcript:**
```
Your question: What do readers say about the pacing of It by Stephen King?

Answer: According to the reviews, one reader says that "this never lags too much"
(review [2]), implying the pacing is generally good. However, another reader criticizes
the book's length ("ELEVEN-FUCKING-HUNDRED pages long", review [4]), suggesting they found
it overly long.
Sources: girl_on_the_train.txt, it.txt, woman_in_the_window.txt

Retrieved from:
• girl_on_the_train.txt
• it.txt
• woman_in_the_window.txt
```

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is *And Then There Were None* good for first-time mystery readers? | Yes — reviewers recommend it as a genre starting point | Says it's a good intro per reviews; notes no review literally says "first-time" | Relevant | Accurate |
| 2 | Common complaints about Agatha Christie's writing style? | Bland/sterile writing, carbon-copy characters, lack of description, dated attitudes | Found "contrived/stiff," "amateurish," "doesn't age well"; missed carbon-copy & dated-attitudes complaints | Partially relevant | Partially accurate |
| 3 | What do readers say about the pacing of *It*? | Too long/bloated (1000+ pages); fans say length is needed | Mixed: cites length complaints but also a wrong-book chunk; answer muddled | Off-target (top hit wrong book) | Partially accurate |
| 4 | Which book do readers compare most to *Gone Girl*? | *The Girl on the Train* ("next Gone Girl") | "The Girl on the Train" | Relevant | Accurate |
| 5 | What do readers say about the ending of *Gone Girl*? | Out-of-scope — should decline | "I don't have enough information on that." | Off-target (no source covers it) | Accurate (correct refusal) |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "What do readers say about the pacing of *It* by Stephen King?"

**What the system returned:** The top-ranked retrieved chunk was from
`girl_on_the_train.txt` ("The pace was slow and sometimes the chapters were just plain
boring"), not from `it.txt`. The generated answer mixed signals — quoting "this never lags
too much" from one *It* review while noting the 1,100-page length from another — and ended
up vaguer than the expected "many readers find it too long but fans defend the length."

**Root cause (tied to a specific pipeline stage):** This is a **retrieval** failure driven
by the **chunking + embedding** stages. My chunks are small (~250 chars) and deliberately
encode a single idea. A chunk whose single idea is "the pace was slow and chapters were
boring" is an extremely strong semantic match for a *pacing* query — regardless of which
book it describes. Because the book title often isn't repeated inside every chunk, the
embedding has no signal to prefer the correct book, so a generic "slow pacing" comment
about *The Girl on the Train* out-ranks the actual *It* reviews. This is the exact
"retrieval returns reviews from the wrong book" risk anticipated in `planning.md`.

**What you would change to fix it:** (1) **Metadata filtering** — detect the book named in
the query and filter retrieval to chunks from that source file (ChromaDB `where` clause on
the `source`/`book_title` metadata I already store). (2) **Prepend the book title to each
chunk's text before embedding** so the title contributes to the embedding and same-book
chunks score higher. (3) **Hybrid search** — combine semantic similarity with a keyword
(BM25) match on the title to break ties toward the right book.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking Strategy and
Retrieval Approach sections of `planning.md` before coding gave the AI tool exact targets
(250-char chunks, 50 overlap, `all-MiniLM-L6-v2`, top-k 5, ChromaDB with source metadata),
so the generated `ingestion.py` and `retrieval.py` matched my design on the first pass
instead of producing generic defaults. The Anticipated Challenges section also paid off:
because I had predicted "retrieval returns reviews from the wrong book," I immediately
recognized the *It* pacing result as that known risk rather than a mysterious bug, and I
already had source metadata attached to every chunk to support the fix.

**One way your implementation diverged from the spec, and why:** The spec said to store an
integer star `rating` per chunk, but I discovered during chunk inspection that some reviews
are rated `4.5 stars`. My original review-parsing regex only matched whole numbers, so
those reviews were silently swallowed (header text leaked into chunks and review counts were
off). I changed the implementation to parse **fractional ratings** and store `rating` as a
float — a divergence from the spec's integer assumption, made because the real data didn't
fit the original assumption.

---

## AI Usage

**Instance 1 — Ingestion and chunking**
- *What I gave the AI:* My `planning.md` Documents and Chunking Strategy sections plus the
  architecture diagram, and the actual format of my review files (title header + `REVIEW N
  (M stars):` markers).
- *What it produced:* `ingestion.py` — loading all 8 files, splitting them into individual
  reviews with rating/number metadata, and chunking with `RecursiveCharacterTextSplitter`
  at `chunk_size=250`, `chunk_overlap=50`.
- *What I changed or overrode:* During the required chunk-inspection step I found a
  `REVIEW 4 (4.5 stars):` header leaking into a chunk — the regex only matched whole-number
  ratings. I directed the fix to handle fractional ratings and store the rating as a float,
  which recovered a previously-dropped review (Then She Was Gone went from 7 → 8 reviews).

**Instance 2 — Embedding, retrieval, and grounded generation**
- *What I gave the AI:* My Retrieval Approach section and the requirement that responses be
  grounded and cite sources.
- *What it produced:* `retrieval.py` (embedding with `all-MiniLM-L6-v2`, ChromaDB storage,
  `retrieve(query, k=5)`) and `generate.py` (the grounding prompt + Groq call).
- *What I changed or overrode:* I had the vector store configured for **cosine distance**
  (rather than the default L2) so scores fall in an interpretable 0–2 range and line up with
  the "good < 0.5" threshold I used to judge retrieval. I also required source attribution
  to be **appended programmatically** from metadata rather than trusting the LLM to cite
  sources, and suppressed citations on out-of-scope refusals.
