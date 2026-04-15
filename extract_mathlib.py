"""
extract_mathlib.py

Extracts lemma names, type signatures, and docstrings from Mathlib4 source files
and writes them to data/mathlib_lemmas.jsonl for indexing.

Usage:
    python extract_mathlib.py --mathlib-path /path/to/mathlib4

Each output line is a JSON object:
    {"name": "Nat.add_comm", "type": "∀ (n m : ℕ), n + m = m + n", "doc": "..."}
"""

import argparse
import json
import os
import re
from pathlib import Path

# Regex patterns for Lean 4 declarations
DECL_PATTERN = re.compile(
    r'(?P<doc>/--.*?-/\s*)?'          # optional docstring
    r'(?:theorem|lemma|def|abbrev)\s+'
    r'(?P<name>[\w\.\']+)\s*'
    r'(?P<rest>[^:=]*?):\s*'
    r'(?P<type>.+?)\s*(?::=|where|by)',
    re.DOTALL
)

DOC_PATTERN = re.compile(r'/--\s*(.*?)\s*-/', re.DOTALL)


def extract_from_file(path: Path) -> list[dict]:
    """Extract lemma entries from a single .lean file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries = []
    for m in DECL_PATTERN.finditer(text):
        name = m.group("name").strip()
        raw_type = m.group("type").strip()
        # Clean up multi-line types
        clean_type = " ".join(raw_type.split())
        doc = ""
        if m.group("doc"):
            doc_match = DOC_PATTERN.search(m.group("doc"))
            if doc_match:
                doc = " ".join(doc_match.group(1).split())
        entries.append({"name": name, "type": clean_type, "doc": doc})
    return entries


def extract_mathlib(mathlib_path: str, output_path: str = "data/mathlib_lemmas.jsonl"):
    """Walk all .lean files in mathlib_path and write JSONL output."""
    root = Path(mathlib_path)
    os.makedirs(Path(output_path).parent, exist_ok=True)

    total = 0
    with open(output_path, "w") as out:
        for lean_file in sorted(root.rglob("*.lean")):
            entries = extract_from_file(lean_file)
            for entry in entries:
                out.write(json.dumps(entry) + "\n")
            total += len(entries)
            if total % 1000 == 0:
                print(f"  Extracted {total} lemmas so far...")

    print(f"Done. {total} lemmas written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mathlib4 lemmas to JSONL")
    parser.add_argument(
        "--mathlib-path",
        required=True,
        help="Path to the root of the Mathlib4 source tree (e.g. ~/.elan/toolchains/...)",
    )
    parser.add_argument(
        "--output",
        default="data/mathlib_lemmas.jsonl",
        help="Output JSONL file path.",
    )
    args = parser.parse_args()
    extract_mathlib(args.mathlib_path, args.output)
