# Vexoo Labs AI Engineer Assignment — Technical Report

---

## Ingestion and Pyramid Approach

Documents are split using a character-level sliding window (2500 chars, 1250-char stride) to preserve contextual overlap between chunks. Each chunk is represented as a four-layer Knowledge Pyramid: raw text, a two-sentence summary, a rule-based domain label, and a distilled layer containing top-5 keywords and a hash-seeded mock TF-IDF vector. This layered structure supports both dense (vector) and sparse (keyword/fuzzy) retrieval without requiring an external embedding model.

---

## GSM8K Training Setup

The training script targets meta-llama/Llama-3.2-1B with LoRA (r=8, alpha=16) applied to q_proj and v_proj, the most parameter-efficient targets for attention-based adaptation. The first 3,000 GSM8K samples are used for training and the next 1,000 for evaluation. Samples are formatted as Question/Answer pairs and truncated or padded to 512 tokens. A FULL_TRAIN flag toggles between real HuggingFace Trainer-based training and a GPU-free simulation mode that exercises all code paths with mock classes.

---

## Key Design Decisions

The mock TF-IDF vectors are seeded deterministically from each chunk MD5 hash, ensuring reproducibility without any fitted vectoriser. Retrieval combines cosine similarity (weight 0.6) with difflib fuzzy matching (weight 0.4) to handle both semantic and lexical overlap. The ReasoningAdapter uses Python eval() with a sanitised expression for numeric math queries, falling back to templated reasoning steps for word problems.

---

## Assumptions

It is assumed that GSM8K is accessible via openai/gsm8k on HuggingFace and that the LLaMA model requires a valid HF token for real training. Simulation mode is the default to support evaluation in CPU-only environments. Stop-word filtering uses a static list rather than an NLP library to eliminate extra dependencies.

---

## Bonus: Reasoning Adapter Architecture

The ReasoningAdapter implements the Strategy + Router pattern. detect_type() acts as the router using regex and keyword scoring, and each handle_*() method is an independent strategy encapsulating domain-specific logic. This design allows any handler to be swapped or mocked in isolation without modifying the routing layer, making it straightforward to plug in real model calls, external APIs, or RAG pipelines in production.
