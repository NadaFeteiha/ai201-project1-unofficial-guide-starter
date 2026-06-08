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


def handle_query(question):
    if not question or not question.strip():
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"])
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide — Book Reviews") as demo:
    gr.Markdown(
        "# The Unofficial Guide — Book Reviews\n"
        "Ask what real readers say about popular mystery, thriller, and horror books. "
        "Answers are grounded in collected Goodreads reviews."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What do readers say about the pacing of It by Stephen King?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

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
