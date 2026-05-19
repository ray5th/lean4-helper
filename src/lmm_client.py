from groq import Groq
from typing import List, Optional


class LMMClient:
    """
    Client for interacting with LLMs via Groq API.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self._client = Groq()

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Sends a chat request to the model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
        )

        # Guard against empty/missing choices from the Groq API.
        # Previously this crashed with IndexError on `response.choices[0]`,
        # which surfaces to callers as an opaque traceback instead of a
        # useful message. Return an empty string so upstream code can
        # treat it as "no suggestion" without blowing up the run.
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content if content is not None else ""

    def generate_proof_steps(self, lean_code: str, goals: List[str], errors: List[str]) -> str:
        """
        Specific helper to generate proof steps based on current Lean state.
        """
        system_prompt = (
            "You are an expert Lean 4 proof assistant. "
            "Your goal is to complete the proof by replacing 'sorry' with valid Lean 4 code. "
            "Use Mathlib theorems where appropriate. "
            "Respond ONLY with the corrected Lean code block."
        )
        prompt = f"""
Current Lean Code:
```lean
{lean_code}
```

Current Proof Goals:
{chr(10).join(goals)}

Lean Errors:
{chr(10).join(errors)}

Please provide the corrected Lean code. Focus on solving the current goals and fixing the errors.
"""
        return self.chat(prompt, system_prompt)
