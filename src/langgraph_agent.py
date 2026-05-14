import os
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _write_file(path: str, code: str) -> None:
    with open(path, "w") as f:
        f.write(code)


def _extract_lean_code(text: str) -> str:
    if "```lean" in text:
        return text.split("```lean")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


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

        return {
            **state,
            "lean_code": code,
            "errors": result["errors"],
            "goals": result["goals"],
            "status": new_status,
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
        new_code = _extract_lean_code(raw)

        if not new_code or new_code.strip() == state["lean_code"].strip():
            print("LLM produced no changes.")
            return {**state, "attempt": state["attempt"] + 1, "status": "failed"}

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
        model_name: str = "qwen3-vl:4b",
        max_retries: int = 5,
        index_dir: str | None = None,
    ):
        self._lean_env = LeanEnvironment(use_mathlib=True)
        self._retriever = MathLibRetriever(index_dir=index_dir)
        self._chain = RAGProofChain(model_name=model_name)
        self._graph = build_graph(self._lean_env, self._retriever, self._chain)
        self._max_retries = max_retries

    def solve_file(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return False

        initial: ProofState = {
            "file_path": file_path,
            "lean_code": "",
            "goals": [],
            "errors": [],
            "attempt": 0,
            "max_retries": self._max_retries,
            "status": "pending",
            "retrieved_lemmas": [],
        }

        final = self._graph.invoke(initial)
        return final["status"] == "success"
