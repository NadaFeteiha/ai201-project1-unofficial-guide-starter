"""
Milestone 6 — Evaluation harness.

Runs all 5 test questions from planning.md through the full pipeline and
prints, for each: the question, the expected answer, the retrieved chunks
(with source + distance), and the system's grounded response.

Accuracy judgments (accurate / partial / inaccurate) are made by reading
this output and recorded in README.md.

Run:
    python evaluate.py
"""

from generate import ask

# Each entry mirrors the Evaluation Plan table in planning.md.
TEST_CASES = [
    {
        "question": "Is And Then There Were None good for first-time mystery readers?",
        "expected": "Yes — multiple reviewers recommend it as a starting point for "
                    "the genre and for Agatha Christie specifically.",
    },
    {
        "question": "What are common complaints about Agatha Christie's writing style?",
        "expected": "Sterile/bland writing, carbon-copy characters that are hard to "
                    "distinguish, lack of descriptive narration, and racist/sexist "
                    "attitudes that have not aged well.",
    },
    {
        "question": "What do readers say about the pacing of It by Stephen King?",
        "expected": "Many readers find the book too long (1000+ pages) and bloated, "
                    "though fans argue the length is needed for character development.",
    },
    {
        "question": "Which book do readers compare most to Gone Girl?",
        "expected": "The Girl on the Train — reviewers say it was marketed as "
                    "'the next Gone Girl'.",
    },
    {
        "question": "What do readers say about the ending of Gone Girl?",
        "expected": "Out-of-scope: Gone Girl is not in the document set. The system "
                    "should decline to answer rather than hallucinate.",
    },
]


def run():
    for i, case in enumerate(TEST_CASES, 1):
        q = case["question"]
        result = ask(q)

        print("=" * 90)
        print(f"Q{i}: {q}")
        print("=" * 90)
        print(f"EXPECTED: {case['expected']}\n")

        print("RETRIEVED CHUNKS:")
        for j, h in enumerate(result["hits"], 1):
            print(f"  #{j}  distance={h['distance']:.3f}  source={h['source']}  "
                  f"rating={h['rating']}")
            print(f"      {h['text'][:160]}{'...' if len(h['text']) > 160 else ''}")

        print("\nSYSTEM RESPONSE:")
        print(result["answer"])
        print()


if __name__ == "__main__":
    run()
