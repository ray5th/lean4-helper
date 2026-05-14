import ollama
from typing import List, Dict, Any, Optional

class LMMClient:
    """
    Client for interacting with local LMMs via Ollama.
    Focuses on Qwen3-VL:4B for high-reasoning tasks.
    """
    
    def __init__(self, model_name: str = "qwen3-vl:4b"):
        self.model_name = model_name

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Sends a chat request to the model.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        messages.append({'role': 'user', 'content': prompt})
        
        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )
        return response['message']['content']

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
