from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


_SYSTEM = (
    "You are an expert Lean 4 proof assistant with deep knowledge of Mathlib. "
    "Your task is to complete the proof by replacing every `sorry` with valid Lean 4 tactic code. "
    "RULES:\n"
    "1. Keep `import Mathlib` exactly as-is at the top. Do NOT add, remove, or change any import lines.\n"
    "2. Do NOT add `open` statements unless they were already in the original code.\n"
    "3. Keep the theorem signature exactly as given — do not change argument names or types.\n"
    "4. Replace `sorry` with valid Lean 4 tactic(s). Prefer simple tactics: `simp`, `omega`, `ring`, `exact`, `apply`.\n"
    "5. Respond ONLY with the complete corrected Lean code inside a single ```lean ... ``` block."
)

_HUMAN = """\
## Current Lean Code
```lean
{lean_code}
```

## Open Proof Goals
{goals}

## Lean Errors
{errors}

## Relevant Mathlib Lemmas
{retrieved_lemmas}

Provide the corrected Lean code that solves all goals and fixes all errors.
"""


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(none retrieved)"
    return "\n".join(
        f"- `{d.metadata.get('name', '?')}`: {d.page_content}" for d in docs
    )


class RAGProofChain:
    """
    LangChain LCEL chain: retrieved context + proof state → corrected Lean code.
    """

    def __init__(self, model_name: str = "qwen3-vl:4b"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        # Disable thinking/chain-of-thought mode (qwen3, gemma3) and cap output
        # so the agent doesn't spend minutes generating reasoning tokens.
        llm = OllamaLLM(
            model=model_name,
            num_predict=1024,           # cap response length
            options={"think": False},   # disable thinking mode (qwen3/gemma3)
        )
        self._chain = prompt | llm | StrOutputParser()

    def generate(
        self,
        lean_code: str,
        goals: List[str],
        errors: List[str],
        retrieved_lemmas: List[Document],
    ) -> str:
        """
        Generate corrected Lean code given the current proof state and retrieved lemmas.
        """
        return self._chain.invoke({
            "lean_code": lean_code,
            "goals": "\n".join(goals) or "(none)",
            "errors": "\n".join(errors) or "(none)",
            "retrieved_lemmas": _format_docs(retrieved_lemmas),
        })
