import io
import os
import sys
import tempfile
import threading

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph_agent import LangGraphAgent
from lean_verifier import LeanEnvironment
from retriever import MathLibRetriever

# ---------------------------------------------------------------------------
# Cached heavyweight components.
#
# `LeanEnvironment` spawns a Lean REPL and loads Mathlib (5-15s cold).
# `MathLibRetriever` loads the ~200MB FAISS index + cross-encoder model.
# Without caching, every solve_proof call paid this cost. We build them once
# on first use and share across requests; LeanEnvironment.verify_proof has
# its own lock so concurrent solves serialize at the Lean step but parallel
# everywhere else.
# ---------------------------------------------------------------------------

_COMPONENT_LOCK = threading.Lock()
_LEAN_ENV: LeanEnvironment | None = None
_RETRIEVER: MathLibRetriever | None = None


def _get_components():
    global _LEAN_ENV, _RETRIEVER
    with _COMPONENT_LOCK:
        if _LEAN_ENV is None:
            print("Initializing Lean environment (one-time)…")
            _LEAN_ENV = LeanEnvironment(use_mathlib=True)
        if _RETRIEVER is None:
            print("Loading FAISS retriever (one-time)…")
            _RETRIEVER = MathLibRetriever()
        return _LEAN_ENV, _RETRIEVER


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
    "openai/gpt-oss-120b",                          # default, strongest open-weight on Groq
    "openai/gpt-oss-20b",                           # fast / cheap GPT-OSS
    "qwen/qwen3-32b",                               # Qwen 3, 32B reasoning model
    "meta-llama/llama-4-scout-17b-16e-instruct",    # Llama 4 MoE
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
    --shadow-sm:   0 1px 2px rgba(26,26,26,0.04), 0 2px 8px rgba(26,26,26,0.04);
    --shadow-md:   0 2px 4px rgba(26,26,26,0.05), 0 8px 24px rgba(26,26,26,0.07);
}

/* ─── Container ─── */
.gradio-container {
    background-color: var(--bg) !important;
    /* soft warm glow behind the header, fading into the cream base */
    background-image: radial-gradient(900px 360px at 50% -120px, #F8E8DD 0%, rgba(248,232,221,0) 70%) !important;
    background-repeat: no-repeat !important;
    color: var(--text) !important;
    color-scheme: light;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 36px 28px 28px !important;
}
footer, .show-api { display: none !important; }

/* ─── Header ─── */
#header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 28px;
    padding: 0 4px;
}
#header .title {
    font-size: 22px;
    font-weight: 650;
    letter-spacing: -0.02em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 11px;
}
#header .title .mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--accent) 0%, #E8946F 100%);
    color: #fff;
    font-size: 16px;
    box-shadow: 0 2px 6px rgba(217,119,87,0.35);
}
#header .chips { display: flex; gap: 6px; }
#header .chip {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-muted);
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 11px;
}

/* ─── Controls bar ─── */
#controls {
    display: flex;
    align-items: flex-end;
    gap: 20px;
    margin-bottom: 18px;
    padding: 14px 18px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-sm);
}
/* strip Gradio's inner block chrome so controls sit flat in the bar */
#controls .block, #controls .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* component labels — small caps over each field */
#controls span[data-testid="block-info"],
#controls label > span:first-child {
    display: inline-block;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted) !important;
    margin-bottom: 5px !important;
}

/* ─── Editor panels ─── */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    max-height: 580px;
}
.panel:focus-within {
    box-shadow: var(--shadow-md);
    border-color: #DCD8CC;
}
.panel-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 10px 14px !important;
    background: var(--panel-tint) !important;
    border-bottom: 1px solid var(--border-soft) !important;
    gap: 8px !important;
    min-height: 46px;
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
    box-shadow: 0 0 0 3px var(--accent-soft);
    display: inline-block;
}
.panel-header .panel-actions {
    display: flex;
    gap: 6px;
    align-items: center;
}

/* ─── Buttons ─── */
.btn button {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    padding: 5px 13px !important;
    font-size: 12px !important;
    font-weight: 550 !important;
    min-width: 0 !important;
    box-shadow: 0 1px 2px rgba(26,26,26,0.04) !important;
    transition: all 0.15s ease !important;
    cursor: pointer;
}
.btn button:hover {
    background: var(--panel-tint) !important;
    border-color: #CFCBBE !important;
}
.btn button:active { transform: translateY(1px); }
.btn button:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}
.btn-primary button {
    background: var(--accent) !important;
    color: white !important;
    border: 1px solid var(--accent) !important;
    box-shadow: 0 1px 3px rgba(217,119,87,0.4) !important;
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
    max-height: 520px;
    overflow: auto;
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
.cm-scroller { max-height: 520px; overflow: auto !important; }

/* readable text in editors regardless of theme/state */
.cm-editor, .cm-editor .cm-content, .cm-editor .cm-line,
.cm-editor .cm-line span, .cm-editor .cm-content * {
    color: #1A1A1A !important;
}

/* thin warm scrollbars in code panes + logs */
.cm-scroller::-webkit-scrollbar,
.gradio-accordion textarea::-webkit-scrollbar { width: 8px; height: 8px; }
.cm-scroller::-webkit-scrollbar-thumb,
.gradio-accordion textarea::-webkit-scrollbar-thumb {
    background: #DDD9CD;
    border-radius: 4px;
}
.cm-scroller::-webkit-scrollbar-thumb:hover,
.gradio-accordion textarea::-webkit-scrollbar-thumb:hover { background: #C9C5B8; }
.cm-scroller::-webkit-scrollbar-track,
.gradio-accordion textarea::-webkit-scrollbar-track { background: transparent; }

/* Remove the harsh default block borders Gradio puts around gr.Code */
.gr-block.gr-code, .gradio-code {
    border: none !important;
    background: transparent !important;
}

/* ─── Spacing between the two panes ─── */
#editor-row { gap: 18px !important; }
@media (max-width: 900px) {
    #editor-row { flex-direction: column !important; }
    #controls { flex-wrap: wrap; }
}

/* ─── Logs accordion ─── */
.gradio-accordion {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 16px !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}
.gradio-accordion > button {
    background: var(--panel-tint) !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 550 !important;
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
    max-height: 240px;
    overflow: auto !important;
}

/* ─── Status bar ─── */
#status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    padding: 11px 16px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 13px;
    color: var(--text);
    box-shadow: var(--shadow-sm);
}
#status-bar .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 11px;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 999px;
    white-space: nowrap;
}
#status-bar .pill.ok    { background: #E8F3EC; color: var(--ok); }
#status-bar .pill.err   { background: #FBE9E7; color: var(--err); }
#status-bar .pill.idle  { background: var(--panel-tint); color: var(--text-muted); }
#status-bar .pill.run   { background: var(--accent-soft); color: var(--accent-hover); }
#status-bar .pill.run::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: statpulse 1.2s ease-in-out infinite;
}
@keyframes statpulse {
    0%, 100% { opacity: 0.35; transform: scale(0.8); }
    50%      { opacity: 1;    transform: scale(1.15); }
}

/* ─── Form chrome (dropdown, slider, key field) ─── */
.gradio-dropdown input,
.gradio-dropdown .wrap {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    font-size: 13px !important;
}
.gradio-dropdown input:focus,
.gradio-dropdown .wrap:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
    outline: none !important;
}
input[type="range"] { accent-color: var(--accent) !important; }
.gradio-slider .head, .gradio-slider input {
    color: var(--text) !important;
    font-size: 13px !important;
    background: var(--panel) !important;
    font-variant-numeric: tabular-nums;
}
input[type="password"], input[type="text"] {
    border-radius: 7px !important;
}
input[type="password"]:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* inputs / textareas — never white on white */
textarea, .gradio-textbox textarea, .gradio-accordion textarea,
input[type="text"], input[type="password"] {
    color: #1A1A1A !important;
    background: var(--panel) !important;
}
::placeholder { color: var(--text-muted) !important; opacity: 1 !important; }

/* dropdown popover — pinned light so options never go white-on-white */
ul[role="listbox"],
.gradio-dropdown ul,
.gradio-dropdown [role="listbox"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    box-shadow: var(--shadow-md) !important;
    color: #1A1A1A !important;
}
ul[role="listbox"] li,
.gradio-dropdown li,
.gradio-dropdown [role="option"] {
    color: #1A1A1A !important;
    background: #FFFFFF !important;
    font-size: 13px !important;
}
ul[role="listbox"] li[aria-selected="true"],
ul[role="listbox"] li:hover,
.gradio-dropdown li:hover,
.gradio-dropdown [role="option"][aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: #1A1A1A !important;
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

        lean_env, retriever = _get_components()

        log_buf = io.StringIO()
        with _capture_stdout(log_buf):
            # fast_model=None: two-stage retry is disabled so every attempt
            # uses the user-selected model and behavior matches the dropdown.
            agent = LangGraphAgent(
                model_name=model_name,
                max_retries=int(max_retries),
                api_key=api_key,
                fast_model=None,
                lean_env=lean_env,
                retriever=retriever,
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
    label = {"ok": "Solved", "err": "Failed", "idle": "Idle", "run": "Solving"}.get(kind, kind)
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
            <div class="chips">
                <span class="chip">Lean 4</span>
                <span class="chip">Mathlib RAG</span>
                <span class="chip">LangGraph</span>
            </div>
        </div>
        """
    )

    # ─── Controls bar ───────────────────────────────────────────────
    with gr.Row(elem_id="controls"):
        model_dropdown = gr.Dropdown(
            choices=ALL_MODELS, value=GROQ_MODELS[0],
            label="Model", scale=2,
        )
        retries_slider = gr.Slider(
            minimum=1, maximum=10, value=5, step=1,
            label="Max retries", scale=1,
        )
        anthropic_key_input = gr.Textbox(
            label="Claude API key (optional)",
            placeholder="sk-ant-… — only for Claude models",
            type="password",
            scale=1,
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

    # ─── Generation-state tracking ──────────────────────────────────
    # Flips True while solve_proof is running so the model-change handler
    # can warn the user that switching mid-flight won't affect the current
    # attempt.
    running_state = gr.State(False)

    def _start_run():
        return True

    def _end_run():
        return False

    def _show_running(model_name: str):
        return _status_html("run", f"Solving with {model_name} — this can take a minute…")

    def _warn_on_model_change(running: bool):
        if running:
            gr.Warning(
                "Generation in progress — switching the model now won't "
                "affect the current attempt and will reset for the next "
                "Solve / Regenerate click."
            )

    model_dropdown.change(_warn_on_model_change, inputs=running_state)

    # ─── Wiring ─────────────────────────────────────────────────────
    solve_btn.click(_start_run, outputs=running_state).then(
        _show_running, inputs=model_dropdown, outputs=status_output,
    ).then(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider, anthropic_key_input],
        outputs=[status_output, code_output, logs_output],
    ).then(_end_run, outputs=running_state)

    regen_btn.click(_start_run, outputs=running_state).then(
        _show_running, inputs=model_dropdown, outputs=status_output,
    ).then(
        solve_proof,
        inputs=[lean_input, model_dropdown, retries_slider, anthropic_key_input],
        outputs=[status_output, code_output, logs_output],
    ).then(_end_run, outputs=running_state)

    reset_btn.click(lambda: EXAMPLE_CODE, outputs=lean_input)
    copy_btn.click(
        fn=None, inputs=code_output,
        js="(code) => { navigator.clipboard.writeText(code); return []; }",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
