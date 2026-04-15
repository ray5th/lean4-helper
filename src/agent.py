"""
agent.py

The core agentic self-correcting loop.

Given a Lean 4 theorem statement, this agent will:
  1. Retrieve relevant Mathlib4 lemmas via FAISS.
  2. Prompt Claude to generate a proof.
  3. Validate the proof with the Lean compiler.
  4. On failure, feed the compiler error back to Claude and retry.
  5. Return the final proof on success or raise after max retries.
"""

import os
from typing import Optional

import anthropic

from .lean_env import LeanResult, run_lean
from .retriever import Retriever

SYSTEM_PROMPT = """\
You are an expert Lean 4 theorem prover with deep knowledge of Mathlib4.
Your task is to write a complete, valid Lean 4 proof for the given theorem.

Rules:
- Output ONLY valid Lean 4 code. No explanation, no markdown, no fences.
- Begin with the necessary imports (e.g., `import Mathlib`).
- Use `theorem` or `lemma` keyword as appropriate.
- Use the provided relevant Mathlib4 lemmas as hints.
- If a previous attempt failed with a compiler error, fix only the reported issue.
"""


def _build_user_message(
    theorem: str,
    context_lemmas: list[dict],
    previous_error: Optional[str] = None,
    previous_code: Optional[str] = None,
) -> str:
    """
    Construct the user message for Claude.
    """
    parts = []

    # Relevant Mathlib4 lemmas from RAG
    if context_lemmas:
        parts.append("## Potentially Relevant Mathlib4 Lemmas\n")
        for lemma in context_lemmas:
            name = lemma.get("name", "?")
            typ = lemma.get("type", "?")
            doc = lemma.get("doc", "")
            parts.append(f"- `{name} : {typ}`" + (f" — {doc}" if doc else ""))
        parts.append("")

    # The theorem to prove
    parts.append(f"## Theorem to Prove\n```lean\n{theorem}\n```\n")

    # Feedback from previous attempt
    if previous_code and previous_error:
        parts.append("## Previous Attempt (FAILED)\n")
        parts.append(f"```lean\n{previous_code}\n```\n")
        parts.append(f"**Compiler Error:**\n```\n{previous_error}\n```\n")
        parts.append("Please fix the error above and output the corrected proof.")
    else:
        parts.append("Please generate a complete Lean 4 proof for the theorem above.")

    return "\n".join(parts)


class ProverAgent:
    """
    Agentic prover that iteratively generates and validates Lean 4 proofs.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model: str = "claude-opus-4-5",
        max_retries: int = 5,
        k_lemmas: int = 5,
        lean_timeout: int = 60,
    ):
        """
        Args:
            retriever:    Optional FAISS Retriever. If None, no RAG context is used.
            model:        Claude model ID to use.
            max_retries:  Maximum number of proof attempts before giving up.
            k_lemmas:     Number of Mathlib lemmas to retrieve per query.
            lean_timeout: Timeout in seconds for the Lean compiler.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Please add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.retriever = retriever
        self.model = model
        self.max_retries = max_retries
        self.k_lemmas = k_lemmas
        self.lean_timeout = lean_timeout

    def prove(self, theorem: str, on_attempt=None) -> tuple[bool, str, int]:
        """
        Attempt to prove the given Lean 4 theorem statement.

        Args:
            theorem:    A Lean 4 theorem/lemma declaration string.
            on_attempt: Optional callback(attempt, code, result) for streaming UI.

        Returns:
            (success, final_code, num_attempts)
        """
        # RAG: retrieve relevant lemmas
        context_lemmas = []
        if self.retriever:
            context_lemmas = self.retriever.retrieve(theorem, k=self.k_lemmas)

        messages = []
        previous_code = None
        previous_error = None

        for attempt in range(1, self.max_retries + 1):
            user_content = _build_user_message(
                theorem=theorem,
                context_lemmas=context_lemmas,
                previous_error=previous_error,
                previous_code=previous_code,
            )

            messages.append({"role": "user", "content": user_content})

            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            generated_code = response.content[0].text.strip()

            # Append assistant response to conversation history
            messages.append({"role": "assistant", "content": generated_code})

            # Validate with Lean compiler
            lean_result: LeanResult = run_lean(generated_code, timeout=self.lean_timeout)

            if on_attempt:
                on_attempt(attempt, generated_code, lean_result)

            if lean_result.success:
                return True, generated_code, attempt

            # Feed error back for next iteration
            previous_code = generated_code
            previous_error = "\n".join(lean_result.errors) or lean_result.output[:500]

        return False, previous_code or "", self.max_retries
