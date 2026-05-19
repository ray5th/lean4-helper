import io
import os
import sys
import tempfile
import threading

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph_agent import LangGraphAgent


# ---------------------------------------------------------------------------
# Thread-safe stdout capture.
#
# The previous implementation wrapped each call in `contextlib.redirect_stdout`,
# which rebinds the process-wide `sys.stdout`. Under concurrent solve_proof
# invocations (e.g. multiple Gradio users), the most-recent rebinding wins for
# every thread, so logs land in the wrong caller's buffer (or get lost when
# one call exits its `with` block and restores stdout while another is mid-run).
#
# We replace it with a thread-local proxy installed once on `sys.stdout`. Each
# call pushes its own StringIO onto its thread-local stack via
# `_capture_stdout()`; prints from other threads continue to land in their own
# buffers (or in the real stdout if they haven't installed one).
# ---------------------------------------------------------------------------

class _ThreadLocalStdout:
    """A `sys.stdout` proxy that dispatches writes to a per-thread buffer."""

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._tls = threading.local()

    def _current(self):
        stack = getattr(self._tls, "stack", None)
        if stack:
            return stack[-1]
        return self._real

    # File-like protocol used by `print()`.
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

    # Stack management ------------------------------------------------------
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
    """Context manager that captures stdout for the current thread only."""

    def __init__(self, buf):
        self._buf = buf

    def __enter__(self):
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
footer, .show-api { display: none !important; }
.gradio-container > .main, .gradio-container .app { padding: 0 !important; gap: 0 !important; }
.block, .gr-block, .form { background: transparent !important; border: none !important; border-radius: 0 !important; box-shadow: none !important; padding: 0 !important; margin: 0 !important; }

/* ----- Top toolbar ----- */
#top-toolbar {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #323233;
    border-bottom: 1px solid #1e1e1e;
    padding: 6px 16px;
    color: #cccccc;
    font-size: 13px;
}
#top-toolbar .logo {
    color: #D97757;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-right: 8px;
}
#top-toolbar .spacer { flex: 1; }

#top-toolbar .gradio-dropdown,
#top-toolbar .gradio-slider {
    min-width: 0 !important;
    margin: 0 !important;
}
#top-toolbar .gradio-dropdown { width: 220px !important; }
#top-toolbar .gradio-slider  { width: 180px !important; }
#top-toolbar label { display: none !important; }

/* ----- Tab header (filename + buttons) ----- */
.tab-header {
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
    background: #2d2d30 !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding: 0 !important;
    margin: 0 !important;
    flex-wrap: nowrap !important;
    min-height: 32px;
}
.tab-header > * { margin: 0 !important; }
.tab-label {
    display: inline-flex;
    align-items: center;
    background: #1e1e1e;
    color: #ffffff;
    padding: 8px 14px;
    font-size: 12px;
    border-right: 1px solid #252526;
    border-top: 2px solid #D97757;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace;
    flex: 0 0 auto;
}
.tab-header-spacer { flex: 1 !important; }

/* Tab-header buttons (Solve / Reset / Regenerate / Copy) */
.tab-btn button {
    background: transparent !important;
    color: #cccccc !important;
    border: 1px solid transparent !important;
    border-radius: 3px !important;
    padding: 3px 10px !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    min-width: 0 !important;
    box-shadow: none !important;
    margin: 0 6px 0 0 !important;
    line-height: 1.4 !important;
    cursor: pointer;
}
.tab-btn button:hover {
    background: #3e3e42 !important;
    color: #ffffff !important;
}
.tab-btn-primary button {
    background: #D97757 !important;
    color: #ffffff !important;
}
.tab-btn-primary button:hover {
    background: #c56a4d !important;
    color: #ffffff !important;
}

/* ----- Code editor ----- */
.cm-editor {
    background: #1e1e1e !important;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.cm-content { color: #d4d4d4 !important; }
.cm-gutters { background: #1e1e1e !important; color: #858585 !important; border-right: 1px solid #252526 !important; }
.cm-activeLine, .cm-activeLineGutter { background: #2a2d2e !important; }
.cm-cursor { border-left-color: #D97757 !important; }

/* Pane container — vertical divider between left/right */
.pane-left  { border-right: 1px solid #1e1e1e; }

/* ----- Logs accordion ----- */
.gradio-accordion {
    background: #252526 !important;
    border-top: 1px solid #1e1e1e !important;
}
.gradio-accordion > button {
    background: #252526 !important;
    color: #cccccc !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    padding: 6px 16px !important;
    border: none !important;
}
.gradio-accordion textarea {
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    border: none !important;
    font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, monospace !important;
    font-size: 12px !important;
}

/* ----- Status bar ----- */
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
    background: rgba(0,0,0,0.18);
    border-radius: 2px;
}

/* ----- Dropdown / slider chrome ----- */
.gradio-dropdown .wrap, .gradio-dropdown input {
    background: #3c3c3c !important;
    color: #d4d4d4 !important;
    border: 1px solid #3e3e42 !important;
    border-radius: 2px !important;
    font-size: 12px !important;
}
input[type="range"] { accent-color: #D97757 !important; }
.gradio-slider .head, .gradio-slider input { color: #d4d4d4 !important; font-size: 12px !important; }
"""


def solve_proof(lean_code: str, model_name: str, max_retries: int):
    if not lean_code.strip():
        return _status_html("idle", "No input"), "", ""

    if not os.environ.get("GROQ_API_KEY"):
        return _status_html("err", "GROQ_API_KEY missing — add it as a Space secret"), "", ""

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False, dir="/tmp")
        tmp_path = tmp.name
        tmp.write(lean_code)
        tmp.close()

        log_buf = io.StringIO()
        # Thread-local stdout capture so concurrent solve_proof calls don't
        # share a single rebound sys.stdout.
        with _capture_stdout(log_buf):
            agent = LangGraphAgent(model_name=model_name, max_retries=int(max_retries))
            result = agent.solve_file_detailed(tmp_path)

        with open(tmp_path) as f:
            final_code = f.read()

        logs = log_buf.getvalue()

        if result["success"]:
            status = _status_html("ok", f"✓ Verified on attempt {result['solved_at_attempt']} / {result['total_attempts']}")
        else:
            status = _status_html("err", f"✗ No proof found after {result['total_attempts']} attempts")

        return status, final_code, logs

    except Exception as exc:
        return _status_html("err", f"✗ Error: {exc}"), "", ""
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _status_html(pill: str, message: str) -> str:
    return f'<div id="status-bar"><span class="pill">{pill}</span><span>{message}</span></div>'


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
    # ─── Top toolbar ──────────────────────────────────────────────
    with gr.Row(elem_id="top-toolbar"):
        gr.HTML('<span class="logo">◆ Lean 4 Proof Assistant</span>')
        gr.HTML('<span class="spacer"></span>')
        model_dropdown = gr.Dropdown(
            choices=GROQ_MODELS,
            value=GROQ_MODELS[0],
            show_label=False,
            container=False,
        )
        retries_slider = gr.Slider(
            minimum=1, maximum=10, value=5, step=1,
            show_label=False,
            container=False,
        )

    # ─── Two-pane split ───────────────────────────────────────────
    with gr.Row(equal_height=True):
        # Left pane: input editor
        with gr.Column(scale=1, elem_classes="pane-left"):
            with gr.Row(elem_classes="tab-header"):
                gr.HTML('<span class="tab-label">theorem.lean</span>')
                gr.HTML('<span class="tab-header-spacer"></span>')
                solve_btn = gr.Button("▶ Solve", elem_classes="tab-btn tab-btn-primary")
                reset_btn = gr.Button("⟲ Reset", elem_classes="tab-btn")
            lean_input = gr.Code(
                language=None,
                value=EXAMPLE_CODE,
                lines=24,
                show_label=False,
                container=False,
            )

        # Right pane: generated proof
        with gr.Column(scale=1):
            with gr.Row(elem_classes="tab-header"):
                gr.HTML('<span class="tab-label">proof.lean</span>')
                gr.HTML('<span class="tab-header-spacer"></span>')
                regen_btn = gr.Button("⟲ Regenerate", elem_classes="tab-btn")
                copy_btn = gr.Button("⎘ Copy", elem_classes="tab-btn")
            code_output = gr.Code(
                language=None,
                interactive=False,
                lines=24,
                show_label=False,
                container=False,
            )

    # ─── Collapsible logs panel ───────────────────────────────────
    with gr.Accordion("▾ Logs", open=False):
        logs_output = gr.Textbox(
            interactive=False,
            lines=8,
            show_label=False,
            container=False,
            placeholder="agent logs will appear here…",
        )

    # ─── Status bar ───────────────────────────────────────────────
    status_output = gr.HTML(_status_html("idle", "ready"))

    # ─── Wiring ───────────────────────────────────────────────────
    solve_btn.click(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider],
        outputs=[status_output, code_output, logs_output],
    )

    regen_btn.click(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider],
        outputs=[status_output, code_output, logs_output],
    )

    reset_btn.click(
        lambda: EXAMPLE_CODE,
        outputs=lean_input,
    )

    copy_btn.click(
        fn=None,
        inputs=code_output,
        js="(code) => { navigator.clipboard.writeText(code); return []; }",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
