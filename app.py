import io
import os
import sys
import tempfile
import threading

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph_agent import LangGraphAgent


# ---------------------------------------------------------------------------
# Thread-safe stdout capture (preserves per-call agent logs under concurrency).
# ---------------------------------------------------------------------------

class _ThreadLocalStdout:
    def __init__(self, real_stdout):
        self._real = real_stdout
        self._tls = threading.local()

    def _current(self):
        stack = getattr(self._tls, "stack", None)
        return stack[-1] if stack else self._real

    def write(self, s):
        return self._current().write(s)

    def flush(self):
        try:
            return self._current().flush()
        except Exception:
            return None

    def isatty(self):
        try:
            return self._real.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._real.fileno()

    def push(self, buf):
        stack = getattr(self._tls, "stack", None)
        if stack is None:
            stack = []
            self._tls.stack = stack
        stack.append(buf)

    def pop(self):
        stack = getattr(self._tls, "stack", None)
        if stack:
            stack.pop()


_STDOUT_PROXY = _ThreadLocalStdout(sys.stdout)
sys.stdout = _STDOUT_PROXY


class _capture_stdout:
    def __init__(self, buf):
        self._buf = buf

    def __enter__(self):
        if sys.stdout is not _STDOUT_PROXY:
            sys.stdout = _STDOUT_PROXY
        _STDOUT_PROXY.push(self._buf)
        return self._buf

    def __exit__(self, exc_type, exc, tb):
        _STDOUT_PROXY.pop()
        return False


GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
]

CLAUDE_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

ALL_MODELS = GROQ_MODELS + CLAUDE_MODELS


def _is_claude(model_name: str) -> bool:
    return model_name.startswith("claude-")

EXAMPLE_CODE = """\
import Mathlib

theorem add_zero_simple (n : ℕ) : n + 0 = n := by
  sorry
"""


# Cream / warm-orange palette inspired by claude.ai
CUSTOM_CSS = """
:root {
    --bg:          #FAF9F5;
    --panel:       #FFFFFF;
    --panel-tint:  #F5F4ED;
    --border:      #E8E5DC;
    --border-soft: #EFEDE4;
    --text:        #1A1A1A;
    --text-muted:  #6B6B6B;
    --accent:      #D97757;
    --accent-hover:#C5694A;
    --accent-soft: #FBE9DF;
    --ok:          #2E7D55;
    --err:         #B85450;
}

/* Container reset */
.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 32px 24px !important;
}
footer, .show-api { display: none !important; }

/* ─── Header ─── */
#header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    padding: 0 4px;
}
#header .title {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
}
#header .title .mark {
    color: var(--accent);
    font-size: 22px;
}
#header .sub {
    font-size: 13px;
    color: var(--text-muted);
}

/* ─── Top control bar (model + retries) ─── */
#controls {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
    padding: 12px 16px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
}
#controls label {
    font-size: 12px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    margin: 0 6px 0 0 !important;
}
#controls .gradio-dropdown,
#controls .gradio-slider { margin: 0 !important; }

/* ─── Editor panels ─── */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.panel-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 10px 14px !important;
    background: var(--panel-tint) !important;
    border-bottom: 1px solid var(--border-soft) !important;
    gap: 8px !important;
    min-height: 44px;
}
.panel-title {
    font-family: 'JetBrains Mono', ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 12px;
    color: var(--text);
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
}
.panel-header .panel-actions {
    display: flex;
    gap: 6px;
    align-items: center;
}

/* ─── Buttons in panel headers ─── */
.btn button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 5px 12px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    min-width: 0 !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
    cursor: pointer;
}
.btn button:hover {
    background: var(--panel-tint) !important;
    border-color: var(--text-muted) !important;
}
.btn-primary button {
    background: var(--accent) !important;
    color: white !important;
    border: 1px solid var(--accent) !important;
}
.btn-primary button:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

/* ─── Code editors ─── */
.cm-editor {
    background: var(--panel) !important;
    font-family: 'JetBrains Mono', ui-monospace, "SF Mono", Menlo, monospace !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}
.cm-content { color: var(--text) !important; padding: 12px 0 !important; }
.cm-gutters {
    background: var(--panel) !important;
    color: #B0AFA3 !important;
    border-right: 1px solid var(--border-soft) !important;
}
.cm-activeLine { background: var(--panel-tint) !important; }
.cm-activeLineGutter { background: var(--panel-tint) !important; color: var(--text) !important; }
.cm-cursor { border-left-color: var(--accent) !important; }
.cm-selectionBackground { background: var(--accent-soft) !important; }

/* Remove the harsh default block borders Gradio puts around gr.Code */
.gr-block.gr-code, .gradio-code {
    border: none !important;
    background: transparent !important;
}

/* ─── Spacing between the two panes ─── */
#editor-row { gap: 16px !important; }

/* ─── Logs accordion ─── */
.gradio-accordion {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 16px !important;
    overflow: hidden;
}
.gradio-accordion > button {
    background: var(--panel-tint) !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    border: none !important;
    text-align: left !important;
}
.gradio-accordion textarea {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: none !important;
    font-family: 'JetBrains Mono', ui-monospace, "SF Mono", Menlo, monospace !important;
    font-size: 12px !important;
    padding: 12px 14px !important;
    line-height: 1.6 !important;
}

/* ─── Status bar ─── */
#status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    padding: 10px 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 13px;
    color: var(--text);
}
#status-bar .pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    border-radius: 999px;
}
#status-bar .pill.ok    { background: #E8F3EC; color: var(--ok); }
#status-bar .pill.err   { background: #FBE9E7; color: var(--err); }
#status-bar .pill.idle  { background: var(--panel-tint); color: var(--text-muted); }

/* ─── Form chrome (dropdown, slider) ─── */
.gradio-dropdown input,
.gradio-dropdown .wrap {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 13px !important;
}
.gradio-dropdown input:focus,
.gradio-dropdown .wrap:focus-within {
    border-color: var(--accent) !important;
    outline: none !important;
}
input[type="range"] { accent-color: var(--accent) !important; }
.gradio-slider .head, .gradio-slider input {
    color: var(--text) !important;
    font-size: 13px !important;
    background: var(--panel) !important;
}
"""


def solve_proof(lean_code: str, model_name: str, max_retries: int, anthropic_api_key: str = ""):
    if not lean_code.strip():
        return _status_html("idle", "No input — paste a Lean 4 theorem on the left."), "", ""

    claude = _is_claude(model_name)
    if claude:
        if not anthropic_api_key or not anthropic_api_key.strip():
            return _status_html("err", "Anthropic API key required for Claude models — paste it above."), "", ""
        api_key = anthropic_api_key.strip()
    else:
        if not os.environ.get("GROQ_API_KEY"):
            return _status_html("err", "GROQ_API_KEY missing — add it as a Space secret."), "", ""
        api_key = None  # ChatGroq picks it up from env

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".lean", mode="w", delete=False, dir="/tmp", encoding="utf-8",
        )
        tmp_path = tmp.name
        tmp.write(lean_code)
        tmp.close()

        log_buf = io.StringIO()
        with _capture_stdout(log_buf):
            agent = LangGraphAgent(
                model_name=model_name,
                max_retries=int(max_retries),
                api_key=api_key,
            )
            result = agent.solve_file_detailed(tmp_path)

        with open(tmp_path, encoding="utf-8") as f:
            final_code = f.read()

        logs = log_buf.getvalue()

        if result["success"]:
            status = _status_html(
                "ok",
                f"Verified on attempt {result['solved_at_attempt']} of {result['total_attempts']}.",
            )
        else:
            status = _status_html("err", f"No proof found after {result['total_attempts']} attempts.")

        return status, final_code, logs

    except Exception as exc:
        return _status_html("err", f"Error: {exc}"), "", ""
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _status_html(kind: str, message: str) -> str:
    label = {"ok": "Solved", "err": "Failed", "idle": "Idle"}.get(kind, kind)
    return (
        f'<div id="status-bar">'
        f'<span class="pill {kind}">{label}</span>'
        f'<span>{message}</span>'
        f'</div>'
    )


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
    # ─── Header ─────────────────────────────────────────────────────
    gr.HTML(
        """
        <div id="header">
            <div class="title"><span class="mark">◆</span> Lean 4 Proof Assistant</div>
            <div class="sub">Mathlib RAG · Groq · LangGraph</div>
        </div>
        """
    )

    # ─── Controls bar ───────────────────────────────────────────────
    with gr.Row(elem_id="controls"):
        model_dropdown = gr.Dropdown(
            choices=ALL_MODELS, value=GROQ_MODELS[0],
            label="Model", show_label=True, container=False, scale=2,
        )
        retries_slider = gr.Slider(
            minimum=1, maximum=10, value=5, step=1,
            label="Max retries", show_label=True, container=False, scale=1,
        )
        anthropic_key_input = gr.Textbox(
            label="Anthropic API key (only for Claude models)",
            placeholder="sk-ant-…",
            type="password",
            show_label=True,
            container=False,
            scale=2,
        )

    # ─── Two editor panes ───────────────────────────────────────────
    with gr.Row(elem_id="editor-row", equal_height=True):
        # Input pane
        with gr.Column(scale=1, elem_classes="panel"):
            with gr.Row(elem_classes="panel-header"):
                gr.HTML('<span class="panel-title">theorem.lean</span>')
                with gr.Row(elem_classes="panel-actions"):
                    solve_btn = gr.Button("Solve", elem_classes="btn btn-primary")
                    reset_btn = gr.Button("Reset", elem_classes="btn")
            lean_input = gr.Code(
                language=None, value=EXAMPLE_CODE,
                lines=22, show_label=False, container=False,
            )

        # Output pane
        with gr.Column(scale=1, elem_classes="panel"):
            with gr.Row(elem_classes="panel-header"):
                gr.HTML('<span class="panel-title">proof.lean</span>')
                with gr.Row(elem_classes="panel-actions"):
                    regen_btn = gr.Button("Regenerate", elem_classes="btn")
                    copy_btn = gr.Button("Copy", elem_classes="btn")
            code_output = gr.Code(
                language=None, interactive=False,
                lines=22, show_label=False, container=False,
            )

    # ─── Status bar ─────────────────────────────────────────────────
    status_output = gr.HTML(_status_html("idle", "Ready. Paste a theorem and click Solve."))

    # ─── Logs (collapsed by default) ────────────────────────────────
    with gr.Accordion("Agent logs", open=False):
        logs_output = gr.Textbox(
            interactive=False, lines=10, show_label=False, container=False,
            placeholder="Agent attempts, retrieval, and verifier output will stream here…",
        )

    # ─── Wiring ─────────────────────────────────────────────────────
    solve_btn.click(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider, anthropic_key_input],
        outputs=[status_output, code_output, logs_output],
    )
    regen_btn.click(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider, anthropic_key_input],
        outputs=[status_output, code_output, logs_output],
    )
    reset_btn.click(lambda: EXAMPLE_CODE, outputs=lean_input)
    copy_btn.click(
        fn=None, inputs=code_output,
        js="(code) => { navigator.clipboard.writeText(code); return []; }",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
