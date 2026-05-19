#!/usr/bin/env python3
"""
MiniF2F benchmark for the LangGraph Lean 4 proof agent.

Metrics reported
----------------
pass@k   : fraction of problems solved within k LLM attempts
           (computed for k = 1, 2, ..., max_retries)
avg_attempts_to_solve : mean attempts used on problems that were solved
avg_time_s            : mean wall-clock seconds per problem

Example
-------
# Quick smoke-test (10 problems, gemma3:12b, 3 retries)
python scripts/benchmark.py --subset 10 --model gemma3:12b --retries 3

# Full valid split with Claude (244 problems, 5 retries)
python scripts/benchmark.py --split valid --model claude-3-5-haiku-20241022 --retries 5

# Ablation: no RAG
python scripts/benchmark.py --subset 50 --no-rag --model gemma3:12b
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from langgraph_agent import LangGraphAgent

# ---------------------------------------------------------------------------
# MiniF2F loading
# ---------------------------------------------------------------------------

_DATASET_CANDIDATES = [
    ("cat-searcher/minif2f-lean4",       "formal_statement"),
]

# HuggingFace split name aliases (MiniF2F uses "validation" not "valid")
_SPLIT_ALIASES = {"valid": "validation", "val": "validation"}


def _ensure_import_and_sorry(code: str) -> str:
    if "import Mathlib" not in code:
        code = "import Mathlib\n\n" + code
    # If proof body is missing or is just whitespace after :=, add sorry
    if ":= by" in code and "sorry" not in code:
        code = code.rstrip() + "\n  sorry\n"
    elif ":=" in code and "sorry" not in code and "by" not in code:
        code = code.rstrip() + " by\n  sorry\n"
    return code


def load_minif2f(split: str = "valid", max_problems: int | None = None):
    from datasets import load_dataset

    hf_split = _SPLIT_ALIASES.get(split, split)

    for dataset_name, stmt_field in _DATASET_CANDIDATES:
        try:
            ds = load_dataset(dataset_name, split=hf_split)
            print(f"Loaded '{dataset_name}' ({split} split): {len(ds)} problems")

            # Normalise to list[dict] with keys: name, lean_code
            rows = []
            for i, row in enumerate(ds):
                name = row.get("name") or row.get("id") or row.get("problem_name") or f"problem_{i}"
                code = None
                for f in [stmt_field, "lean_code", "statement", "code", "formal_statement"]:
                    if f in row and row[f]:
                        code = _ensure_import_and_sorry(row[f])
                        break
                if code is None:
                    continue
                rows.append({"name": name, "lean_code": code})

            if max_problems:
                rows = rows[:max_problems]
            print(f"Using {len(rows)} problems after filtering.")
            return rows

        except Exception as e:
            print(f"  Could not load '{dataset_name}': {e}")

    raise RuntimeError(
        "Could not load MiniF2F from any known HuggingFace source.\n"
        "Try: pip install datasets  and check your internet connection."
    )


def load_local_problems(problems_dir: str, max_problems: int | None = None):
    """Load `.lean` files from a directory as a list of {name, lean_code} dicts."""
    root = Path(problems_dir)
    if not root.is_dir():
        raise RuntimeError(f"Problems directory not found: {problems_dir}")

    files = sorted(root.glob("*.lean"))
    if max_problems:
        files = files[:max_problems]

    rows = []
    for path in files:
        code = path.read_text(encoding="utf-8")
        if "sorry" not in code:
            # Skip files that are already complete proofs.
            continue
        rows.append({"name": path.stem, "lean_code": code})

    print(f"Loaded {len(rows)} local problem(s) with sorry placeholders.")
    return rows


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k(results: list[dict], k: int) -> float:
    """Fraction of problems solved within the first k attempts."""
    if not results:
        return 0.0
    solved = sum(
        1 for r in results
        if r["success"] and r["solved_at_attempt"] <= k
    )
    return solved / len(results)


# ---------------------------------------------------------------------------
# Single-problem runner
# ---------------------------------------------------------------------------

def run_one(agent: LangGraphAgent, name: str, lean_code: str, verbose: bool) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", prefix=f"bench_{name[:20]}_", delete=False
    ) as f:
        f.write(lean_code)
        tmp = f.name

    try:
        t0 = time.time()
        detail = agent.solve_file_detailed(tmp)
        elapsed = round(time.time() - t0, 2)
    finally:
        # Restore original sorry so the temp file doesn't leak a partial proof
        try:
            os.unlink(tmp)
        except OSError:
            pass

    result = {
        "name": name,
        "success": detail["success"],
        "solved_at_attempt": detail["solved_at_attempt"],
        "total_attempts": detail["total_attempts"],
        "time_s": elapsed,
    }

    if verbose:
        status = "PASS" if result["success"] else "FAIL"
        print(
            f"  [{status}] {name:<50} "
            f"attempt={result['solved_at_attempt'] or '-':>2}  "
            f"time={elapsed:>6.1f}s"
        )
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], max_retries: int, model: str, no_rag: bool):
    n = len(results)
    solved = [r for r in results if r["success"]]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Model       : {model}")
    print(f"  RAG         : {'disabled' if no_rag else 'enabled'}")
    print(f"  Problems    : {n}")
    print(f"  Max retries : {max_retries}")
    print()
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for k in range(1, max_retries + 1):
        pct = pass_at_k(results, k) * 100
        print(f"  {'pass@' + str(k):<25} {pct:>9.1f}%")
    print()
    if solved:
        avg_att = sum(r["solved_at_attempt"] for r in solved) / len(solved)
        avg_t   = sum(r["time_s"] for r in results) / n
        print(f"  {'avg attempts (solved)':<25} {avg_att:>10.2f}")
        print(f"  {'avg time/problem (s)':<25} {avg_t:>10.1f}")
    print("=" * 60)


def write_csv(results: list[dict], path: str):
    fieldnames = ["name", "success", "solved_at_attempt", "total_attempts", "time_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the Lean proof agent on MiniF2F and report pass@k metrics."
    )
    parser.add_argument("--split",   default="valid",        help="Dataset split: valid (=validation) | test")
    parser.add_argument("--subset",  type=int, default=None, help="Use only first N problems")
    parser.add_argument("--model",   default="llama-3.3-70b-versatile", help="Groq / Claude model ID")
    parser.add_argument("--retries", type=int, default=5,    help="Max LLM attempts per problem")
    parser.add_argument("--no-rag",  action="store_true",    help="Disable RAG retrieval (ablation)")
    parser.add_argument("--index-dir", default=None,         help="Path to pre-built FAISS index")
    parser.add_argument("--output",  default="benchmark_results.csv", help="CSV output path")
    parser.add_argument("--verbose", action="store_true",    help="Print per-problem results")
    parser.add_argument("--api-key", default=None,           help="API key for the chosen provider (Anthropic for Claude models). Falls back to ANTHROPIC_API_KEY / GROQ_API_KEY env.")
    parser.add_argument("--problems-dir", default=None,      help="Use local .lean files in this directory instead of MiniF2F. Each file is one problem.")
    args = parser.parse_args()

    if args.problems_dir:
        print(f"Loading local problems from {args.problems_dir}…")
        problems = load_local_problems(args.problems_dir, max_problems=args.subset)
    else:
        print(f"Loading MiniF2F ({args.split} split)…")
        problems = load_minif2f(split=args.split, max_problems=args.subset)

    print(f"Initialising agent (model={args.model}, retries={args.retries})…")
    agent = LangGraphAgent(
        model_name=args.model,
        max_retries=args.retries,
        index_dir=args.index_dir,
        api_key=args.api_key,
    )

    if args.no_rag:
        # Monkey-patch retriever to return empty results
        agent._retriever.retrieve = lambda query: []

    results = []
    print(f"\nRunning {len(problems)} problems…\n")
    for i, prob in enumerate(problems, 1):
        print(f"[{i:>3}/{len(problems)}] {prob['name'][:60]}")
        r = run_one(agent, prob["name"], prob["lean_code"], verbose=args.verbose)
        results.append(r)

        # Rolling summary every 10 problems
        if i % 10 == 0:
            p1 = pass_at_k(results, 1) * 100
            pk = pass_at_k(results, args.retries) * 100
            print(f"  → Rolling pass@1={p1:.1f}%  pass@{args.retries}={pk:.1f}%  ({i}/{len(problems)} done)\n")

    print_summary(results, args.retries, args.model, args.no_rag)
    write_csv(results, args.output)


if __name__ == "__main__":
    main()
