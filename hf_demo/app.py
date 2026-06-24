"""Gradio demo for RAG Document Q&A - Hugging Face Spaces."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Lazy pipeline initialisation (avoids heavy imports at module load time)
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        # These imports happen inside HF Space at runtime
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        from rag_qa.config import Settings
        from rag_qa.pipeline import RAGPipeline

        settings = Settings(
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            qdrant_url="",  # in-memory
            reranker_enabled=True,
        )
        _pipeline = RAGPipeline(settings)
    return _pipeline


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


def upload_file(file_obj) -> str:
    """Ingest an uploaded file and return a status message."""
    if file_obj is None:
        return "No file selected."

    pipeline = get_pipeline()
    file_path = file_obj.name
    filename = Path(file_path).name

    import asyncio

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(pipeline.ingest(file_path, filename))
        loop.close()
        return (
            f"Ingested '{result['filename']}'\n"
            f"  Chunks created : {result['chunks']}\n"
            f"  Vectors stored : {result['vectors_stored']}\n"
            f"  Doc ID         : {result['doc_id']}"
        )
    except Exception as exc:
        return f"Error: {exc}"


def list_docs() -> str:
    """Return a formatted list of ingested documents."""
    try:
        pipeline = get_pipeline()
        docs = pipeline.list_documents()
        if not docs:
            return "No documents ingested yet."
        lines = [f"- {d['filename']} ({d['chunk_count']} chunks, id={d['doc_id'][:8]}...)" for d in docs]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def ask_question(question: str, top_k: int, history: list) -> tuple[list, list]:
    """Query the pipeline and update the chatbot history."""
    if not question.strip():
        return history, history

    if not os.environ.get("GROQ_API_KEY"):
        bot_msg = "GROQ_API_KEY is not set. Please add it as a Space secret."
        history = history + [(question, bot_msg)]
        return history, history

    try:
        pipeline = get_pipeline()
        import asyncio

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(pipeline.query(question, top_k=int(top_k)))
        loop.close()

        answer = result["answer"]
        if result["sources"]:
            sources_text = "\n\n**Sources:**\n" + "\n".join(
                f"- {s['filename']} (chunk {s['chunk_index']}, score={s['score']:.3f})"
                for s in result["sources"][:3]
            )
            answer += sources_text

        history = history + [(question, answer)]
    except Exception as exc:
        history = history + [(question, f"Error: {exc}")]

    return history, history


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="RAG Document Q&A",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 900px; margin: auto; }",
) as demo:
    gr.Markdown(
        """
        # RAG Document Q&A
        Upload PDF, DOCX, or TXT documents and ask questions about them.
        Powered by **Groq Llama-3.3-70b** + **MiniLM** embeddings + **Qdrant** vector search.
        """
    )

    chat_history = gr.State([])

    with gr.Tabs():
        # ------------------------------------------------------------------
        # Tab 1: Upload
        # ------------------------------------------------------------------
        with gr.Tab("Upload Documents"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Select file (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                    )
                    upload_btn = gr.Button("Upload & Ingest", variant="primary")
                with gr.Column(scale=3):
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        lines=6,
                        interactive=False,
                    )

            gr.Markdown("---")
            refresh_btn = gr.Button("Refresh Document List")
            docs_list = gr.Textbox(label="Ingested Documents", lines=5, interactive=False)

            upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_status])
            refresh_btn.click(fn=list_docs, inputs=[], outputs=[docs_list])

        # ------------------------------------------------------------------
        # Tab 2: Ask
        # ------------------------------------------------------------------
        with gr.Tab("Ask Questions"):
            chatbot = gr.Chatbot(label="Conversation", height=420)

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    label="Question",
                    scale=5,
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-K results",
                    scale=1,
                )

            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            ask_btn.click(
                fn=ask_question,
                inputs=[question_box, top_k_slider, chat_history],
                outputs=[chatbot, chat_history],
            )
            question_box.submit(
                fn=ask_question,
                inputs=[question_box, top_k_slider, chat_history],
                outputs=[chatbot, chat_history],
            )
            clear_btn.click(lambda: ([], []), outputs=[chatbot, chat_history])

    gr.Markdown(
        """
        ---
        **Setup:** Add `GROQ_API_KEY` as a Space secret (Settings -> Secrets).
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
