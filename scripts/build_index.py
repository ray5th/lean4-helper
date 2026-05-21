#!/usr/bin/env python3
"""
One-time script to build the FAISS + BM25 index from Mathlib4 source files.
Run this before using the LangGraph agent for the first time:

    python scripts/build_index.py

Optional flags:
    --mathlib-root PATH   Path to Mathlib4 source (auto-detected if omitted)
    --max-files N         Limit to first N .lean files (useful for quick testing)
    --index-dir PATH      Where to save the index (default: data/mathlib_index)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retriever import MathLibRetriever


def main():
    parser = argparse.ArgumentParser(description="Build Mathlib FAISS index.")
    parser.add_argument("--mathlib-root", default=None, help="Path to Mathlib4 source root")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of .lean files processed")
    parser.add_argument("--index-dir", default=None, help="Directory to save the index")
    args = parser.parse_args()

    retriever = MathLibRetriever(index_dir=args.index_dir)

    if retriever.is_index_built() and not args.max_files:
        print(f"Index already exists at {retriever.index_dir}. Delete it to rebuild.")
        return

    retriever.build(mathlib_root=args.mathlib_root, max_files=args.max_files)
    print("Done. Index is ready for use.")


if __name__ == "__main__":
    main()
