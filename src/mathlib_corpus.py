import os
import re
import glob
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


# Regex to capture optional docstring + declaration line
_DECL_PATTERN = re.compile(
    r'(?:/--\s*(.*?)\s*-/)?\s*'               # optional /-- docstring -/
    r'(?:@\[.*?\]\s*)*'                        # optional attributes
    r'(theorem|lemma|def|noncomputable def)\s+'
    r'(\S+)\s*'                                # declaration name
    r'(.*?)\s*:=',                             # everything up to :=
    re.DOTALL,
)


def _find_mathlib_root() -> Optional[str]:
    """
    Returns the path to the Mathlib4 source directory, searching common locations.
    """
    candidates = [
        # Lake package cache (used by lean-interact's TempRequireProject)
        os.path.expanduser("~/.elan/toolchains"),
        os.path.expanduser("~/.cache/mathlib"),
        # Nix / Homebrew Lean setups
        "/usr/local/lib/lean",
        "/opt/homebrew/lib/lean",
    ]

    # Also check for a .lake/packages directory next to the current file
    here = Path(__file__).resolve().parent.parent
    lake_pkg = here / ".lake" / "packages" / "mathlib" / "Mathlib"
    if lake_pkg.exists():
        return str(lake_pkg.parent)

    for root in candidates:
        if not os.path.isdir(root):
            continue
        # Walk up to 4 levels looking for a Mathlib directory
        for dirpath, dirnames, _ in os.walk(root):
            depth = dirpath.replace(root, "").count(os.sep)
            if depth > 4:
                dirnames.clear()
                continue
            if "Mathlib" in dirnames:
                return dirpath
    return None


def _parse_lean_file(path: str) -> List[Document]:
    """
    Extracts theorem/lemma/def declarations from a single .lean file.
    Returns a list of Documents with page_content = "<name> : <signature>".
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    docs = []
    for match in _DECL_PATTERN.finditer(text):
        docstring = (match.group(1) or "").strip()
        kind = match.group(2)
        name = match.group(3)
        signature = re.sub(r'\s+', ' ', match.group(4)).strip()

        content = f"{name} : {signature}"
        if docstring:
            content = f"{docstring}\n{content}"

        line = text[: match.start()].count("\n") + 1
        docs.append(Document(
            page_content=content,
            metadata={"kind": kind, "name": name, "file": path, "line": line},
        ))
    return docs


class MathLibCorpus:
    """
    Extracts LangChain Documents from Mathlib4 source files on disk.
    """

    def __init__(self, mathlib_root: Optional[str] = None):
        self.mathlib_root = mathlib_root or _find_mathlib_root()

    def extract(self, max_files: Optional[int] = None) -> List[Document]:
        """
        Walks Mathlib source files and extracts declaration Documents.

        Args:
            max_files: If set, stop after processing this many .lean files
                       (useful for quick tests).

        Returns:
            List of LangChain Documents, one per declaration found.
        """
        if not self.mathlib_root:
            raise RuntimeError(
                "Could not locate Mathlib4 source. "
                "Pass mathlib_root explicitly or run `lake exe cache get` first."
            )

        pattern = os.path.join(self.mathlib_root, "**", "*.lean")
        files = glob.glob(pattern, recursive=True)
        if max_files:
            files = files[:max_files]

        docs: List[Document] = []
        for path in files:
            docs.extend(_parse_lean_file(path))

        return docs
