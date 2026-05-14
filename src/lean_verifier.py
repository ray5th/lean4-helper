from typing import Dict, Any, List

from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject, LeanRequire


class LeanEnvironment:
    """
    Manages the Lean REPL environment for verifying Lean 4 proofs.
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

    def verify_proof(self, lean_code: str) -> Dict[str, Any]:
        """
        Executes a block of Lean code and verifies if it is a correct proof.
        
        Args:
            lean_code (str): The full Lean 4 code string containing imports, theorem statement, and proof.
            
        Returns:
            dict: A dictionary containing the status, errors (if any), and goals (if open sorries remain).
        """
        response = self.server.run(Command(cmd=lean_code))

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
