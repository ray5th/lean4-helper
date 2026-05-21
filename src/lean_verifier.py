import threading
from typing import Any, Dict

from lean_interact import Command, LeanREPLConfig, LeanServer, TempRequireProject


class LeanEnvironment:
    """
    Manages the Lean REPL environment for verifying Lean 4 proofs.

    Safe to share across threads: `verify_proof` serializes concurrent calls
    behind a lock, because the underlying LeanServer is a single-process
    stdin/stdout pipe and would otherwise interleave commands.
    """

    def __init__(self, use_mathlib: bool = True, lean_version: str = "v4.8.0"):
        """
        Initializes the Lean environment.

        Args:
            use_mathlib (bool): If True, configures a TempRequireProject with Mathlib.
                                This may take a while to build on the first run.
            lean_version (str): The Lean 4 version to use. Default is v4.8.0.
        """
        self.lean_version = lean_version
        self.use_mathlib = use_mathlib

        if self.use_mathlib:
            # We use TempRequireProject with mathlib as specified in lean_interact documentation
            project = TempRequireProject(
                lean_version=self.lean_version,
                require="mathlib"
            )
            self.config = LeanREPLConfig(project=project)
        else:
            self.config = LeanREPLConfig(lean_version=self.lean_version)

        self.server = LeanServer(self.config)
        self._lock = threading.Lock()

    def verify_proof(self, lean_code: str) -> Dict[str, Any]:
        """
        Executes a block of Lean code and verifies if it is a correct proof.

        Args:
            lean_code (str): The full Lean 4 code string containing imports, theorem statement, and proof.

        Returns:
            dict: A dictionary containing the status, errors (if any), and goals (if open sorries remain).
        """
        # Guard: empty/whitespace input would crash `Command(cmd=...)` with a
        # pydantic ValidationError (min_length=1). Return a clean failure.
        if not lean_code or not lean_code.strip():
            return {
                "status": "failure",
                "errors": ["empty Lean code: nothing to verify"],
                "goals": [],
                "env": None,
            }

        # Guard: LeanServer can raise on disconnect / crash / OOM. Surface as a
        # failure result so the agent's retry loop continues instead of dying.
        # The lock serializes concurrent requests since LeanServer is single-process.
        try:
            with self._lock:
                response = self.server.run(Command(cmd=lean_code))
        except Exception as exc:
            return {
                "status": "failure",
                "errors": [f"Lean server error: {exc}"],
                "goals": [],
                "env": None,
            }

        errors = []
        goals = []

        # Check for error or warning messages
        if hasattr(response, 'messages') and response.messages:
            for msg in response.messages:
                if msg.severity in ['error', 'warning']:
                    # E.g., 'declaration uses 'sorry'' is a warning, but we might want to capture it
                    errors.append(msg.data)

        # Check for open goals (sorries)
        if hasattr(response, 'sorries') and response.sorries:
            for sorry in response.sorries:
                if sorry.goal:
                    goals.append(sorry.goal)

        is_success = len(errors) == 0 and len(goals) == 0

        return {
            "status": "success" if is_success else "failure",
            "errors": errors,
            "goals": goals,
            "env": getattr(response, "env", None)
        }
