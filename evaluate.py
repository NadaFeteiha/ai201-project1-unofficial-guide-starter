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
        "question": "Is And Then There Were None a good first mystery book to read?",
        "expected": "Yes — reviewers say it is a good place to start the genre and a "
                    "good Agatha Christie book to begin with.",
    },
    {
        "question": "What do readers complain about in Agatha Christie's writing?",
        "expected": "Plain/sterile writing, similar characters that are hard to tell "
                    "apart, little description, and old-fashioned attitudes that have "
                    "not aged well.",
    },
    {
        "question": "What do readers say about the pacing of It by Stephen King?",
        "expected": "Many readers say it is too long (1000+ pages) and bloated, though "
                    "fans say the length is needed for the characters.",
    },
    {
        "question": "Which book do readers compare most to Gone Girl?",
        "expected": "The Girl on the Train — reviewers say it was called "
                    "'the next Gone Girl'.",
    },
    {
        "question": "What do readers say about the ending of Gone Girl?",
        "expected": "Out-of-scope: Gone Girl is not in the documents. The system should "
                    "refuse to answer instead of making something up.",
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
