## Brief Non-Technical MVP: AI Lean 4 Proof Assistant

### Project goal

Build an AI assistant that helps users write correct **Lean 4 mathematical proofs**.

The user provides a theorem. The system searches Lean’s Mathlib library, finds useful related theorems, asks an AI model to write a proof, checks the proof using Lean, and automatically retries if the proof fails.

The key idea is:

```text
AI suggests the proof.
Lean verifies the proof.
Only Lean decides if the answer is correct.
```

---

## MVP flow

```text
User enters theorem
        ↓
Lean tooling reads the theorem and current proof goal
        ↓
System retrieves useful Mathlib lemmas
        ↓
System reranks the lemmas and keeps the best ones
        ↓
LangChain organizes the prompt and retrieval flow
        ↓
Local LLM generates a Lean proof
        ↓
Lean checks the proof
        ↓
If correct → return final proof
If wrong → send Lean error back to the AI and retry
```

---

## Simple explanation

This project is like an **AI coding assistant for mathematical proofs**.

But unlike a normal chatbot, it does not just guess. It uses Lean itself to check whether the proof is truly valid.

The assistant works in three stages:

```text
1. Search
Find useful existing Mathlib theorems.

2. Generate
Ask the AI model to write a proof using those theorems.

3. Verify
Run the proof through Lean and retry if Lean finds an error.
```

---

## Main MVP components

### 1. Lean tooling

Use **LeanInteract** to connect Python with Lean 4.

LeanInteract lets Python interact with Lean through the Lean REPL, which means your program can send Lean code, inspect proof goals, and receive Lean feedback programmatically. ([GitHub][1])

In simple terms:

```text
LeanInteract lets the AI system talk to Lean.
```

Instead of only seeing an error like:

```text
type mismatch
```

the system can see more useful information like:

```text
Current goal:
n : Nat
⊢ n = n
```

That makes the retry much smarter.

---

### 2. Retrieve Mathlib lemmas

The system searches Mathlib for useful theorems.

For example, if the theorem is about even numbers, the retriever may find lemmas related to:

```text
Even
Odd
Nat.even_or_odd
parity
multiplication
powers
```

This is the “retrieve” part.

Use:

```text
FAISS for fast semantic search
BM25 / keyword search for exact Lean names
```

FAISS is useful for quickly finding similar text from a large collection of embedded documents, while keyword search helps when exact theorem names matter.

---

### 3. Rerank retrieved lemmas

Retrieval alone is not enough.

The system may retrieve 50 possible Mathlib lemmas, but many may be weak or irrelevant. So we add a **reranker**.

The improved flow is:

```text
Retrieve top 50 lemmas
        ↓
Rerank them based on the current theorem and proof goal
        ↓
Keep the best 8–12 lemmas
        ↓
Send only those to the AI model
```

A Cross-Encoder reranker is useful here because it scores each query-document pair more carefully than the first-pass retriever. Sentence Transformers describes this as a common retrieve-and-rerank setup where fast retrieval finds candidates and the Cross-Encoder reranks them for better relevance. ([SentenceTransformers][2])

In simple terms:

```text
Retrieve finds possible useful lemmas.
Rerank chooses the best ones.
```

This improves the quality of the context given to the AI.

---

### 4. LangChain role

Use **LangChain** for orchestration, not for proof correctness.

LangChain can help organize:

```text
retrieval
reranking
prompt templates
document formatting
LLM calls
retry history
```

LangChain has retrieval utilities such as contextual compression, where retrieved documents can be filtered or shortened before sending them to the model. ([LangChain][3])

But LangChain does **not** make Lean proofs correct by itself.

The correctness still comes from:

```text
Lean compiler
Lean proof state
Mathlib retrieval quality
LLM proof generation
retry loop
```

So the right positioning is:

```text
LangChain manages the workflow.
Lean verifies the truth.
```

---

### 5. Local LLM first

For the MVP, use a local model first.

Example:

```text
Ollama + Qwen Coder
```

Why local first?

```text
No API cost
Easy to test many times
Good for debugging the pipeline
Private and simple for development
```

Later, once the workflow works, replace the local model with Claude.

The architecture stays the same.

---

### 6. Lean verification and retry

After the AI generates a proof, the system sends it to Lean.

If Lean accepts the proof:

```text
Success → return proof
```

If Lean rejects the proof:

```text
Failure → capture Lean error → add error to prompt → retry
```

Example:

```text
Attempt 1:
AI generates proof.
Lean says: unknown theorem name.

Attempt 2:
AI sees the error and tries a different lemma.
Lean says: type mismatch.

Attempt 3:
AI fixes the proof.
Lean accepts it.
```

This is the most important part of the MVP.

---

## Final MVP architecture

```text
User Theorem
     │
     ▼
LeanInteract
Get theorem goal and Lean feedback
     │
     ▼
Hybrid Retrieval
FAISS semantic search + keyword search
     │
     ▼
Reranker
Keep only the most useful Mathlib lemmas
     │
     ▼
LangChain Prompt Flow
Organize theorem, lemmas, history, and Lean errors
     │
     ▼
Local LLM
Generate Lean proof
     │
     ▼
Lean Checker
Verify proof
     │
     ├── Success → Return final proof
     │
     └── Failure → Retry with Lean error
```

---

## Libraries to use

| Purpose                      | Library                                           |
| ---------------------------- | ------------------------------------------------- |
| Interact with Lean           | `LeanInteract`                                    |
| Semantic retrieval           | `sentence-transformers`                           |
| Vector search                | `faiss-cpu`                                       |
| Keyword search               | `rank-bm25`                                       |
| Reranking                    | `sentence-transformers` CrossEncoder or FlashRank |
| Workflow orchestration       | `LangChain`                                       |
| More advanced workflow graph | `LangGraph`                                       |
| Local model                  | `Ollama`                                          |
| Future stronger model        | Claude API                                        |

LangGraph is useful later if you want the proof process to behave like a clear state machine: retrieve → generate → verify → retry → success/failure. LangGraph is designed for stateful agent workflows, which fits this kind of multi-step proof loop. ([Emergent Mind][4])

---

## Non-technical value proposition

This MVP creates an AI assistant that helps people write mathematically verified Lean proofs.

It does not only generate text. It checks every answer with Lean, learns from errors, and keeps trying until it produces a valid proof.

The project demonstrates:

```text
AI-assisted formal reasoning
verified code generation
search over mathematical knowledge
automatic error correction
LLM + compiler feedback loops
```

---

## One-line MVP summary

```text
An AI-powered Lean 4 proof assistant that searches Mathlib, reranks useful lemmas, generates a proof with a local LLM, checks it using Lean tooling, and automatically retries using Lean errors until the proof is verified.
```

[1]: https://github.com/augustepoiroux/LeanInteract?utm_source=chatgpt.com "LeanInteract: A Python Interface for Lean 4"
[2]: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html?utm_source=chatgpt.com "Retrieve & Re-Rank Pipeline"
[3]: https://www.langchain.com/blog/improving-document-retrieval-with-contextual-compression?utm_source=chatgpt.com "Improving Document Retrieval with Contextual Compression"
[4]: https://www.emergentmind.com/topics/langgraph?utm_source=chatgpt.com "LangGraph: Modular LLM Agent Orchestration"
