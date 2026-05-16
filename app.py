import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph_agent import LangGraphAgent

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
]

EXAMPLE_CODE = """\
import Mathlib

theorem add_zero_simple (n : ℕ) : n + 0 = n := by
  sorry
"""


def solve_proof(lean_code: str, model_name: str, max_retries: int) -> tuple[str, str, str]:
    if not lean_code.strip():
        return "Please enter some Lean 4 code.", "", ""

    if not os.environ.get("GROQ_API_KEY"):
        return "GROQ_API_KEY is not set. Add it as a Space secret.", "", ""

    tmp = tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False, dir="/tmp")
    try:
        tmp.write(lean_code)
        tmp.close()

        log_buf = io.StringIO()
        with redirect_stdout(log_buf):
            agent = LangGraphAgent(model_name=model_name, max_retries=int(max_retries))
            result = agent.solve_file_detailed(tmp.name)

        with open(tmp.name) as f:
            final_code = f.read()

        logs = log_buf.getvalue()

        if result["success"]:
            status = f"Proof verified on attempt {result['solved_at_attempt']} of {result['total_attempts']}."
        else:
            status = f"Could not find a proof after {result['total_attempts']} attempts."

        return status, final_code, logs

    except Exception as exc:
        return f"Error: {exc}", "", ""
    finally:
        os.unlink(tmp.name)


with gr.Blocks(title="Lean 4 Proof Assistant") as demo:
    gr.Markdown(
        "# Lean 4 Proof Assistant\n"
        "Paste Lean 4 code containing `sorry` placeholders. "
        "The agent will use Mathlib RAG + an LLM to complete the proof and verify it with the Lean REPL.\n\n"
        "> **Note:** Proof attempts can take 1–5 minutes. Please be patient."
    )

    with gr.Row():
        with gr.Column(scale=1):
            lean_input = gr.Code(
                label="Lean 4 Code",
                language=None,
                value=EXAMPLE_CODE,
                lines=18,
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=GROQ_MODELS,
                    value=GROQ_MODELS[0],
                    label="Model",
                )
                retries_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Max Retries",
                )
            submit_btn = gr.Button("Solve Proof", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Status", interactive=False, lines=2)
            code_output = gr.Code(
                label="Completed Proof",
                language=None,
                interactive=False,
                lines=14,
            )
            logs_output = gr.Textbox(label="Agent Logs", interactive=False, lines=8)

    submit_btn.click(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider],
        outputs=[status_output, code_output, logs_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
