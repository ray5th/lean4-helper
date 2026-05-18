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


CUSTOM_CSS = """
/* ----- Reset Gradio chrome ----- */
.gradio-container {
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

footer { display: none !important; }
.show-api { display: none !important; }

/* ----- Top title bar (VSCode menu strip) ----- */
#title-bar {
    background: #323233;
    border-bottom: 1px solid #1e1e1e;
    padding: 6px 16px;
    font-size: 13px;
    color: #cccccc;
    display: flex;
    align-items: center;
    gap: 16px;
}

#title-bar .logo {
    color: #D97757;
    font-weight: 700;
    letter-spacing: 0.3px;
}

#title-bar .subtle {
    color: #858585;
    font-size: 12px;
}

/* ----- Tab bar above editors ----- */
.editor-tab {
    background: #2d2d30 !important;
    border: none !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding: 0 !important;
    margin: 0 !important;
}

.editor-tab .tab-label {
    display: inline-block;
    background: #1e1e1e;
    color: #ffffff;
    padding: 8px 14px;
    font-size: 12px;
    border-right: 1px solid #252526;
    border-top: 2px solid #D97757;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace;
}

.editor-tab .tab-label-inactive {
    background: #2d2d30;
    color: #969696;
    border-top: 2px solid transparent;
}

/* ----- Code editor styling (Monaco/CodeMirror overrides) ----- */
.cm-editor {
    background: #1e1e1e !important;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

.cm-content { color: #d4d4d4 !important; }
.cm-gutters { background: #1e1e1e !important; color: #858585 !important; border-right: 1px solid #252526 !important; }
.cm-activeLine { background: #2a2d2e !important; }
.cm-activeLineGutter { background: #2a2d2e !important; color: #c6c6c6 !important; }
.cm-cursor { border-left-color: #D97757 !important; }

/* Code block wrapper */
.block.gr-block, .gradio-container .block {
    background: #1e1e1e !important;
    border: 1px solid #2d2d30 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* Hide Gradio's default labels above code blocks (we use our own tabs) */
.gr-code label, .code-container > label {
    display: none !important;
}

/* ----- Sidebar / control panel ----- */
.control-panel {
    background: #252526 !important;
    border-left: 1px solid #1e1e1e !important;
    padding: 16px !important;
}

/* ----- Labels ----- */
label, .label-wrap {
    color: #cccccc !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
}

/* ----- Dropdown ----- */
.gradio-dropdown, select, .wrap.dropdown {
    background: #3c3c3c !important;
    color: #d4d4d4 !important;
    border: 1px solid #3e3e42 !important;
    border-radius: 2px !important;
    font-size: 12px !important;
}

.gradio-dropdown input { color: #d4d4d4 !important; }

/* ----- Slider ----- */
input[type="range"] { accent-color: #D97757 !important; }
.gradio-slider .head { color: #d4d4d4 !important; }

/* ----- Buttons ----- */
button.lg, button.primary, .gr-button-primary {
    background: #D97757 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    box-shadow: none !important;
    transition: background 0.15s ease;
}

button.lg:hover, button.primary:hover, .gr-button-primary:hover {
    background: #c56a4d !important;
}

button.lg:disabled {
    background: #555 !important;
    cursor: not-allowed !important;
}

/* ----- Textbox / output ----- */
textarea, .gradio-textbox textarea, .gr-text-input {
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    border: 1px solid #2d2d30 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace !important;
    font-size: 12px !important;
}

/* ----- Status bar (VSCode bottom strip) ----- */
#status-bar {
    background: #D97757;
    color: #ffffff;
    padding: 4px 16px;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace;
    display: flex;
    align-items: center;
    gap: 16px;
}

#status-bar .pill {
    padding: 1px 8px;
    background: rgba(0,0,0,0.15);
    border-radius: 2px;
}

/* ----- Layout tweaks ----- */
.gr-row, .row { gap: 0 !important; }
.gr-column, .column { gap: 8px !important; }
.gradio-container > div { gap: 0 !important; }
"""


def solve_proof(lean_code: str, model_name: str, max_retries: int) -> tuple[str, str, str]:
    if not lean_code.strip():
        return "● No input.", "", ""

    if not os.environ.get("GROQ_API_KEY"):
        return "● GROQ_API_KEY missing — add it as a Space secret.", "", ""

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
            status = f"✓ Verified on attempt {result['solved_at_attempt']} / {result['total_attempts']}"
        else:
            status = f"✗ No proof found after {result['total_attempts']} attempts"

        return status, final_code, logs

    except Exception as exc:
        return f"✗ Error: {exc}", "", ""
    finally:
        os.unlink(tmp.name)


with gr.Blocks(
    title="Lean 4 Proof Assistant",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        """
        <div id="title-bar">
            <span class="logo">◆ Lean 4 Proof Assistant</span>
            <span class="subtle">— Mathlib RAG · Groq · LangGraph</span>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        # Left pane: editor
        with gr.Column(scale=3):
            gr.HTML('<div class="editor-tab"><span class="tab-label">theorem.lean</span></div>')
            lean_input = gr.Code(
                language=None,
                value=EXAMPLE_CODE,
                lines=22,
                show_label=False,
            )

        # Right pane: output
        with gr.Column(scale=3):
            gr.HTML('<div class="editor-tab"><span class="tab-label">proof.lean</span><span class="tab-label tab-label-inactive">logs</span></div>')
            code_output = gr.Code(
                language=None,
                interactive=False,
                lines=14,
                show_label=False,
            )
            logs_output = gr.Textbox(
                interactive=False,
                lines=7,
                show_label=False,
                placeholder="agent logs will stream here…",
            )

        # Far right: controls
        with gr.Column(scale=1, elem_classes="control-panel"):
            gr.HTML('<div style="color:#858585;font-size:11px;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:12px;">Configuration</div>')
            model_dropdown = gr.Dropdown(
                choices=GROQ_MODELS,
                value=GROQ_MODELS[0],
                label="Model",
                show_label=True,
            )
            retries_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Max retries",
                show_label=True,
            )
            submit_btn = gr.Button("Solve Proof", variant="primary", size="lg")
            gr.HTML('<div style="color:#858585;font-size:11px;margin-top:12px;line-height:1.5;">Proofs typically take 1–5 minutes. The agent will retry on failure.</div>')

    status_output = gr.HTML('<div id="status-bar"><span class="pill">idle</span><span>ready</span></div>')

    def wrap_status(status: str, code: str, logs: str):
        pill = "ok" if status.startswith("✓") else ("err" if status.startswith("✗") else "...")
        html = f'<div id="status-bar"><span class="pill">{pill}</span><span>{status}</span></div>'
        return html, code, logs

    submit_btn.click(
        lambda code, model, retries: wrap_status(*solve_proof(code, model, retries)),
        inputs=[lean_input, model_dropdown, retries_slider],
        outputs=[status_output, code_output, logs_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
