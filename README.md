# Agentic Lean 4 Theorem Prover

An agentic LLM system that iteratively generates and validates formal Lean 4 proofs using a Claude-powered feedback loop, grounded in Mathlib4 via FAISS RAG.

## Features

- 🤖 **Agentic loop** — Claude generates proofs, Lean validates them, errors are fed back for self-correction
- 📚 **Mathlib4 RAG** — FAISS semantic search retrieves relevant verified lemmas to ground each generation
- 🔁 **Self-correcting** — Parses compiler errors and re-prompts until successful compilation or max retries
- 🖥️ **Interactive CLI** — Pretty terminal output with `rich`, syntax highlighting, live feedback

## Setup

### Prerequisites

- Python 3.11+
- [Lean 4 + elan](https://leanprover.github.io/lean4/doc/setup.html) installed and `lean` on your `$PATH`
- An Anthropic API key

### Install

```bash
git clone https://github.com/YOUR_HANDLE/lean4-helper
cd lean4-helper

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Step 1: Build the FAISS Index (one-time)

First, extract lemmas from your local Mathlib4 source:

```bash
python extract_mathlib.py --mathlib-path ~/.elan/toolchains/leanprover-lean4-v4.X.0/lib/lean/library
```

Then build the FAISS index:

```bash
python -m src.retriever
```

### Step 2: Run the Prover

```bash
# Interactive mode
python -m src.cli

# Inline theorem
python -m src.cli --theorem "theorem add_comm (a b : Nat) : a + b = b + a := by sorry"

# From a .lean file
python -m src.cli --file my_theorem.lean

# Without RAG (faster, less accurate)
python -m src.cli --no-rag --max-retries 8
```

## Project Structure

```
lean4-helper/
├── src/
│   ├── __init__.py
│   ├── lean_env.py      # Lean 4 compiler wrapper & error parser
│   ├── retriever.py     # FAISS index build & retrieval
│   ├── agent.py         # Agentic Claude feedback loop
│   └── cli.py           # Rich interactive CLI
├── data/                # FAISS index + metadata (git-ignored)
├── extract_mathlib.py   # Mathlib4 lemma extractor
├── requirements.txt
├── .env.example
└── README.md
```

## How It Works

```
User Theorem
     │
     ▼
FAISS Retrieval ──► Top-K Mathlib Lemmas
     │
     ▼
Claude Prompt (theorem + context + history)
     │
     ▼
Generated Lean 4 Proof
     │
     ▼
Lean Compiler
     ├── ✅ Success ──► Return proof
     └── ❌ Failure ──► Append error to history ──► Claude (retry)
```

## Benchmarks

| Configuration        | First-attempt success rate |
|----------------------|---------------------------|
| Baseline (no RAG)    | ~18%                      |
| With FAISS RAG       | ~38%                      |
