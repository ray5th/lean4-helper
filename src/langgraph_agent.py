import os
import re
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from lean_verifier import LeanEnvironment
from rag_chain import RAGProofChain
from retriever import MathLibRetriever


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ProofState(TypedDict):
    file_path: str
    lean_code: str
    goals: List[str]
    errors: List[str]
    attempt: int
    max_retries: int
    status: str          # "pending" | "success" | "failed"
    retrieved_lemmas: list
    solved_at_attempt: int  # 0 = unsolved, else the attempt number that succeeded


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    # Lean source uses ∀ ∃ ℕ ↑ etc. Force UTF-8 so non-UTF-8 default locales
    # (e.g. C/POSIX inside minimal Docker images) don't corrupt or crash on
    # read.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, code: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)


# Match an opening fence tagged `lean` (or `lean4`, `Lean`, etc.) followed by a
# newline or whitespace — but NOT something like ```leanish that would
# accidentally consume the "ish" into the code body.
_LEAN_FENCE_RE = re.compile(r"```\s*lean[0-9]*\s*\n", re.IGNORECASE)


def _extract_lean_code(text: str) -> str:
    """
    Extract the Lean code block from an LLM response.

    Handles:
      - ```lean\n...\n```         (canonical)
      - ```lean4\n...\n```        (some LLMs)
      - ```Lean\n...\n```         (case variation)
      - ```\n...\n```             (no language tag)
      - plain text without fences (returned as-is)
    """
    m = _LEAN_FENCE_RE.search(text)
    if m:
        rest = text[m.end():]
        return rest.split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _sanitize_imports(code: str) -> str:
    """
    LLMs often hallucinate Lean import paths. This function strips all `import`
    lines from the generated code and replaces them with `import Mathlib`, which
    is the correct single import for any Mathlib-based proof.
    """
    lines = code.splitlines()
    non_import_lines = [l for l in lines if not l.strip().startswith("import ")]
    return "import Mathlib\n\n" + "\n".join(non_import_lines).lstrip()


# A declaration keyword must be followed by whitespace or end-of-line so that
# `examplelike` (false positive) doesn't match and `theorem\n` (no trailing
# space) does match. `theorem:` and `theorem(` are not valid Lean syntax
# (the declaration name must come first), so we don't allow those either.
_THEOREM_KEYWORD_RE = re.compile(r"^\s*(?:example|theorem|lemma|def)(?:\s|$)")


def _count_theorem_blocks(code: str) -> int:
    return sum(
        1 for line in code.splitlines()
        if _THEOREM_KEYWORD_RE.match(line)
    )


def make_verify_node(lean_env: LeanEnvironment):
    def verify_node(state: ProofState) -> ProofState:
        print(f"\n--- Attempt {state['attempt'] + 1} / {state['max_retries']} ---")
        code = _read_file(state["file_path"])
        result = lean_env.verify_proof(code)

        new_status = "success" if result["status"] == "success" else "pending"
        if new_status == "success":
            print("Proof verified successfully!")
        else:
            print(
                f"Verification failed. "
                f"Errors: {len(result['errors'])}, Goals: {len(result['goals'])}"
            )

        solved_at = state["attempt"] + 1 if new_status == "success" else state["solved_at_attempt"]
        return {
            **state,
            "lean_code": code,
            "errors": result["errors"],
            "goals": result["goals"],
            "status": new_status,
            "solved_at_attempt": solved_at,
        }
    return verify_node


def make_retrieve_node(retriever: MathLibRetriever):
    def retrieve_node(state: ProofState) -> ProofState:
        query = " ".join(state["goals"] + state["errors"])
        print("Retrieving relevant Mathlib lemmas…")
        lemmas = retriever.retrieve(query)
        print(f"  Retrieved {len(lemmas)} lemma(s).")
        return {**state, "retrieved_lemmas": lemmas}
    return retrieve_node


def make_generate_node(chain: RAGProofChain):
    def generate_node(state: ProofState) -> ProofState:
        print("Generating proof with LLM…")
        raw = chain.generate(
            lean_code=state["lean_code"],
            goals=state["goals"],
            errors=state["errors"],
            retrieved_lemmas=state["retrieved_lemmas"],
        )
        new_code = _sanitize_imports(_extract_lean_code(raw))

        if not new_code or new_code.strip() == state["lean_code"].strip():
            print("LLM produced no changes.")
            return {**state, "attempt": state["attempt"] + 1, "status": "failed"}

        original_blocks = _count_theorem_blocks(state["lean_code"])
        generated_blocks = _count_theorem_blocks(new_code)
        if original_blocks > 0 and generated_blocks < original_blocks:
            print(
                f"LLM dropped theorem statements "
                f"({generated_blocks} of {original_blocks} preserved) — rejecting."
            )
            return {**state, "attempt": state["attempt"] + 1}

        _write_file(state["file_path"], new_code)
        print("File updated.")
        return {
            **state,
            "lean_code": new_code,
            "attempt": state["attempt"] + 1,
        }
    return generate_node


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def should_continue(state: ProofState) -> str:
    if state["status"] == "success":
        return END
    if state["attempt"] >= state["max_retries"]:
        return END
    return "retrieve"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(lean_env: LeanEnvironment, retriever: MathLibRetriever, chain: RAGProofChain):
    g = StateGraph(ProofState)

    g.add_node("verify", make_verify_node(lean_env))
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("generate", make_generate_node(chain))

    g.set_entry_point("verify")
    g.add_conditional_edges("verify", should_continue, {"retrieve": "retrieve", END: END})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")

    return g.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class LangGraphAgent:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        max_retries: int = 5,
        index_dir: str | None = None,
    ):
        self._lean_env = LeanEnvironment(use_mathlib=True)
        self._retriever = MathLibRetriever(index_dir=index_dir)
        self._chain = RAGProofChain(model_name=model_name)
        self._graph = build_graph(self._lean_env, self._retriever, self._chain)
        self._max_retries = max_retries

    def solve_file(self, file_path: str) -> bool:
        return self.solve_file_detailed(file_path)["success"]

    def solve_file_detailed(self, file_path: str) -> dict:
        """Returns {"success": bool, "solved_at_attempt": int, "total_attempts": int}."""
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return {"success": False, "solved_at_attempt": 0, "total_attempts": 0}

        initial: ProofState = {
            "file_path": file_path,
            "lean_code": "",
            "goals": [],
            "errors": [],
            "attempt": 0,
            "max_retries": self._max_retries,
            "status": "pending",
            "retrieved_lemmas": [],
            "solved_at_attempt": 0,
        }

        final = self._graph.invoke(initial)
        return {
            "success": final["status"] == "success",
            "solved_at_attempt": final["solved_at_attempt"],
            "total_attempts": final["attempt"],
        }
