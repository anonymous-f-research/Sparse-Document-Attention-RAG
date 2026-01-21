# Sparse (Document)-Attention RAG under Corpus Knowledge Poisoning Attacks

## Overview
This repository contains the official code for the paper "Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention”.

This repository implements an experimental pipeline for **Retrieval-Augmented Generation (RAG)** under **Corpus Knowledge Poisoning**. It evaluates **Sparse Document Attention RAG (SDAG)** against a causal-attention-based RAG (CARG) baseline and additional defense baselines, including **SDAG + baseline integration**.

SDAG is a **block-sparse attention mechanism** that **disallows cross-attention between retrieved documents**, mitigating harmful cross-document interactions when the retrieved set contains conflicting or adversarial evidence. SDAG is applied as a **minimal inference-time change** (attention mask only), with **no fine-tuning** and no architectural changes required.

---

## Repository layout

```
.
├── data/                         # Optional datasets, sampled queries, outputs
├── src/
│   └── pipeline/
│       ├── attack/               # Adversarial document generation & selection (Random/Near/Far)
│       ├── defenses/             # Defense baselines and integrations
│       ├── retrieval/            # Dense / sparse / hybrid retrieval
│       ├── sdag/                 # Sparse Document Attention (block-sparse masks)
│       ├── metrics/              # ACC/ASR and detailed retrieval-ground-truth statistics
│       ├── prompts/              # Prompt templates
│       ├── config.py             # Central configuration (paper-aligned knobs)
│       └── main.py               # End-to-end pipeline entrypoint
└── README.md
```

---

## RAG, Threat model, and SDAG Evaluation

### RAG formulation
Given a question, a retriever  retrieves a set of k documents from a corpus.
A generator then produces an answer based on the retrieved documents.
After tokenization, each retrieved document corresponds to a token block.

### Threat model: Corpus Knowledge Poisoning
The attacker injects adversarial documents crafted to steer the generator toward a false target answer.

Two evaluation settings are supported:
- **In-corpus**: adversarial documents are injected into the corpus and may or may not be retrieved.
- **In-set**: adversarial documents are assumed to appear in the retrieved set, isolating retrieval effects.

### Adversarial document Attack Strategy
Adversarial documents can be selected using:
- **Random**
- **Near** - closest adversarial document to the benign-docs retrieved set's centroid in embedding space
- **Far** - furthest adversarial document to the benign-docs retrieved set's centroid in embedding space

---

## What this repository provides

The pipeline supports:
1. Query loading / sampling (Natural Questions, HotpotQA, TriviaQA, or CSV).
2. Retrieval using **dense**, **sparse (BM25)**, or **hybrid** retrievers.
3. **Corpus poisoning attacks**:
   - adversarial document injection (single and multiple adversarial document insertion).
   - document corruption variants.
  The adversarial documents can be loaded from a csv file or generated using the RAG LLM.
4. **Defenses**:
   - none (CARG baseline)
   - SDAG (document-isolated generation)
   - RAGDefender
   - Discern-and-Answer
   - integrations (e.g., SDAG + baseline defenses)
5. Evaluation using **Accuracy (ACC)** and **Attack Success Rate (ASR)**, plus detailed retrieval–ground-truth analyses.

---

## Installation

Example dependencies:

```bash
pip install torch transformers sentence-transformers faiss-cpu pyserini tqdm numpy
```

---

## Quickstart

Run with default configuration:

```bash
python -m src.pipeline.main
```

Run with a JSON config override:

```bash
python -m src.pipeline.main path/to/config.json
```

---

## Configuration (paper-aligned)

Important parameters:
- `TOP_K`: number of retrieved documents \(k\).
- `RETRIEVER_BACKEND`: dense / sparse / hybrid.
- `ATTACK_VARIANT`: malicious document injection or corruption.
- `MALICIOUS_DOC_SELECTION_STRATEGY`: random / near / far.
- `DEFENSE_BACKEND`: none (CARG), sdag, ragdefender, discern_and_answer.
- `DOC_NEIGHBORS_K`: number of neighboring documents allowed to attend (0 = strict SDAG).

Example override:

```json
{
  "TOP_K": [5, 10],
  "ADD_ATTACK_IN_RANK": [1, 3],
  "RETRIEVER_BACKEND": "dense",
  "MALICIOUS_DOC_SELECTION_STRATEGY": "random",
  "RANKER_MODEL_NAME": "intfloat/e5-large-v2",
  "DEVICE": "cuda:",
  "TEMPERATURE": 0.1
}
```

---

## Outputs

For each `(top_k, attacker_pos)` configuration, the pipeline produces:
- **CSV**: per-query outputs (answers, retrieved docs, match flags).
- **JSON**: aggregated metrics including:
  - ACC / ASR statistics,
  - retrieval–ground-truth bucket analyses,
  - iso vs no-iso overlap behavior,
  - a full **run configuration snapshot** for reproducibility.

Output directories are created automatically.

---


