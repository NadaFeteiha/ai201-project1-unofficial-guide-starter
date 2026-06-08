"""
Milestone 5 — Query interface (Gradio).

A user types a question, the system retrieves relevant reader reviews,
and returns a grounded answer plus the source documents it drew from.

Run:
    python app.py
Then open http://localhost:7860
"""

import gradio as gr

from generate import ask


def _format_chunks(hits):
    """Render retrieved chunks as Markdown: distance, book, rating, and text.

    Lower distance = closer match. Anything under 0.5 is a strong match.
    """
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"**#{i} — distance `{h['distance']:.3f}`** · "
            f"*{h['book_title']}* · {h['rating']}★ · `{h['source']}`\n\n"
            f"> {h['text']}"
        )
    return "\n\n---\n\n".join(lines)


def handle_query(question, k):
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    result = ask(question, k=int(k))
    sources = "\n".join(f"• {s}" for s in result["sources"])
    chunks_md = _format_chunks(result["hits"])
    return result["answer"], sources, chunks_md


with gr.Blocks(title="Book Reviews") as demo:
    gr.Markdown(
        "# Book Reviews depend on Goodreads\n"
        "Ask what real readers say about popular mystery, thriller, and horror books. "
        "Answers are grounded in collected Goodreads reviews."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What do readers say about the pacing of It by Stephen King?",
    )
    k_slider = gr.Slider(
        minimum=1, maximum=10, value=5, step=1,
        label="Chunks to retrieve (top-k)",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    chunks = gr.Markdown(label="Retrieved chunks (with distance scores)")

    btn.click(handle_query, inputs=[inp, k_slider], outputs=[answer, sources, chunks])
    inp.submit(handle_query, inputs=[inp, k_slider], outputs=[answer, sources, chunks])

    gr.Examples(
        examples=[
            "Is And Then There Were None good for first-time mystery readers?",
            "What are common complaints about Agatha Christie's writing style?",
            "What do readers say about the ending of Gone Girl?",
        ],
        inputs=inp,
    )


if __name__ == "__main__":
    demo.launch()
