#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from langgraph_agent import LangGraphAgent


def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph Lean Proof Agent on a file.")
    parser.add_argument("file", help="Path to the .lean file to solve")
    parser.add_argument("--model", default="qwen3-vl:4b", help="Ollama model name")
    parser.add_argument("--retries", type=int, default=5, help="Max retries")
    parser.add_argument("--index-dir", default=None, help="Path to pre-built FAISS index directory")

    args = parser.parse_args()

    print(f"Starting LangGraph Proof Agent with model: {args.model}")
    agent = LangGraphAgent(
        model_name=args.model,
        max_retries=args.retries,
        index_dir=args.index_dir,
    )

    success = agent.solve_file(args.file)

    if success:
        print("\nSuccess! The proof has been verified.")
    else:
        print("\nFailed to verify the proof.")


if __name__ == "__main__":
    main()
