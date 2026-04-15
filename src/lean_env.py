"""
lean_env.py

Wrapper around the Lean 4 compiler. Sends Lean 4 code to the compiler,
captures stdout/stderr, and parses the output to determine success or failure.
"""

import subprocess
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class LeanResult:
    success: bool
    output: str
    errors: list[str]
    raw: str


def run_lean(code: str, timeout: int = 60) -> LeanResult:
    """
    Write `code` to a temporary .lean file, run `lean` on it, and parse
    the compiler output.

    Args:
        code:    Full Lean 4 source code string to validate.
        timeout: Maximum seconds to wait for the compiler.

    Returns:
        A LeanResult with success flag, cleaned output, and parsed errors.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["lean", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        raw = result.stdout + result.stderr
        errors = _parse_errors(raw)
        success = result.returncode == 0 and len(errors) == 0
        return LeanResult(success=success, output=raw.strip(), errors=errors, raw=raw)
    except FileNotFoundError:
        return LeanResult(
            success=False,
            output="",
            errors=["Lean 4 not found. Please install it via elan."],
            raw="",
        )
    except subprocess.TimeoutExpired:
        return LeanResult(
            success=False,
            output="",
            errors=[f"Lean compiler timed out after {timeout}s."],
            raw="",
        )
    finally:
        os.unlink(tmp_path)


def _parse_errors(output: str) -> list[str]:
    """
    Extract human-readable error messages from Lean 4 compiler output.
    Lean 4 errors look like:
        /path/to/file.lean:10:5: error: unknown identifier 'foo'
    """
    pattern = re.compile(r".*?:\d+:\d+:\s*(error|warning):\s*(.+)")
    errors = []
    for line in output.splitlines():
        m = pattern.match(line)
        if m and m.group(1) == "error":
            errors.append(m.group(2).strip())
    return errors
