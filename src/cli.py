"""
cli.py

Interactive CLI for the Agentic Lean 4 Theorem Prover.

Usage:
    python -m src.cli
    python -m src.cli --theorem "theorem add_comm (a b : Nat) : a + b = b + a := by sorry"
    python -m src.cli --file theorems.lean
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich import print as rprint
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich import box

from .agent import ProverAgent
from .lean_env import LeanResult

# Load .env for ANTHROPIC_API_KEY
load_dotenv()

console = Console()

BANNER = """
[bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║   🧮  Agentic Lean 4 Theorem Prover              ║[/bold cyan]
[bold cyan]║   Powered by Claude + FAISS + Mathlib4 RAG       ║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]
"""


def on_attempt_callback(attempt: int, code: str, result: LeanResult):
    """Rich-formatted callback for each proof attempt."""
    console.print(Rule(f"[bold]Attempt {attempt}[/bold]", style="dim"))
    console.print(Syntax(code, "lean", theme="monokai", line_numbers=True))

    if result.success:
        console.print(Panel("[bold green]✅ Lean accepted this proof![/bold green]", box=box.ROUNDED))
    else:
        errors_text = "\n".join(result.errors) if result.errors else result.output[:300]
        console.print(
            Panel(
                f"[bold red]❌ Compiler rejected proof:[/bold red]\n\n[red]{errors_text}[/red]",
                box=box.ROUNDED,
            )
        )


def run_prover(theorem: str, use_rag: bool = True, max_retries: int = 5):
    """
    Main entry point: set up the agent and run it on the given theorem.
    """
    console.print(BANNER)
    console.print(Panel(f"[bold yellow]Theorem to prove:[/bold yellow]\n\n{theorem}", box=box.ROUNDED))

    # Optionally initialize retriever
    retriever = None
    if use_rag:
        index_path = "data/mathlib.index"
        meta_path = "data/mathlib_meta.pkl"
        if os.path.exists(index_path) and os.path.exists(meta_path):
            from .retriever import Retriever
            console.print("[dim]Loading FAISS index for Mathlib4 RAG...[/dim]")
            retriever = Retriever(index_path=index_path, meta_path=meta_path)
            console.print("[green]✓ FAISS index loaded.[/green]\n")
        else:
            console.print(
                "[yellow]⚠ FAISS index not found. Running without RAG context.[/yellow]\n"
                "[dim]Run `python -m src.retriever` to build the index first.[/dim]\n"
            )

    agent = ProverAgent(retriever=retriever, max_retries=max_retries)

    console.print(Rule("[bold]Starting Proof Search[/bold]", style="cyan"))
    success, final_code, num_attempts = agent.prove(theorem, on_attempt=on_attempt_callback)

    console.print(Rule(style="cyan"))
    if success:
        console.print(
            Panel(
                f"[bold green]🎉 Proof found in {num_attempts} attempt(s)![/bold green]",
                box=box.DOUBLE,
            )
        )
        console.print("\n[bold]Final Proof:[/bold]")
        console.print(Syntax(final_code, "lean", theme="monokai", line_numbers=True))
    else:
        console.print(
            Panel(
                f"[bold red]💀 Failed to find a proof after {num_attempts} attempts.[/bold red]\n"
                "[dim]Try increasing --max-retries or refining the theorem statement.[/dim]",
                box=box.DOUBLE,
            )
        )
        if final_code:
            console.print("\n[bold]Last Generated Code:[/bold]")
            console.print(Syntax(final_code, "lean", theme="monokai", line_numbers=True))


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Lean 4 Theorem Prover powered by Claude + FAISS + Mathlib4"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--theorem",
        type=str,
        help="A Lean 4 theorem declaration string to prove.",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to a .lean file containing the theorem to prove.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable Mathlib4 RAG retrieval.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of proof attempts (default: 5).",
    )
    args = parser.parse_args()

    if args.file:
        theorem = Path(args.file).read_text().strip()
    elif args.theorem:
        theorem = args.theorem.strip()
    else:
        console.print("[bold cyan]Enter your Lean 4 theorem (end with Ctrl+D on a new line):[/bold cyan]")
        try:
            lines = sys.stdin.read()
            theorem = lines.strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            sys.exit(0)

    if not theorem:
        console.print("[red]No theorem provided. Exiting.[/red]")
        sys.exit(1)

    run_prover(theorem=theorem, use_rag=not args.no_rag, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
