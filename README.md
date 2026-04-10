# Vexoo Labs — AI Engineer Assignment

A complete, production-structured Python project covering document ingestion with Knowledge Pyramids, LoRA-based fine-tuning on GSM8K, and a plug-and-play reasoning adapter.

---

## Folder Structure

```
vexoo_assignment/
├── part1_ingestion/
│   └── ingestion_pipeline.py   # Sliding Window + Knowledge Pyramid + Retrieval
├── part2_training/
│   └── gsm8k_finetune.py       # LoRA SFT on GSM8K (simulation + real mode)
├── bonus/
│   └── reasoning_adapter.py    # Domain-routing ReasoningAdapter
├── README.md
└── report.md
```

---

## Setup

```bash
pip install datasets transformers peft torch tqdm
```

> `difflib`, `re`, `hashlib`, `json` are all Python standard library — no install needed.  
> Python 3.8+ required.

---

## How to Run

### Part 1 — Document Ingestion Pipeline

```bash
cd part1_ingestion
python ingestion_pipeline.py
```

Runs entirely **offline, no GPU**. Uses a built-in ~800-word AI/ML sample document.  
Output: chunk count, Knowledge Pyramid for first 2 chunks, ranked retrieval results for 2 queries.

---

### Part 2 — GSM8K LoRA Fine-Tuning

```bash
cd part2_training
python gsm8k_finetune.py
```

Default:  (simulation mode — **no GPU or internet needed**).

**Simulation mode:**
- If  is installed, loads real GSM8K data; otherwise generates synthetic samples.
- Uses  and  to exercise all code paths without a GPU.
- Simulates 3 epochs with a mock decreasing loss curve and ~30% eval accuracy.

**Real training mode:**
1. Set  at the top of .
2. Ensure a GPU is available and you have HuggingFace access to .
3. Run .

---

### Bonus — ReasoningAdapter

```bash
cd bonus
python reasoning_adapter.py
```

Runs **offline, no GPU**. Demonstrates routing of 4 query types (math, legal, medical, general) with pretty-printed structured responses.

---

## Requirements

| Library | Purpose | Required when |
|---------|---------|---------------|
|  | Load GSM8K dataset | Part 2 (real data) |
|  | AutoTokenizer, Trainer | Part 2 FULL_TRAIN=True |
|  | LoRA via LoraConfig | Part 2 FULL_TRAIN=True |
|  | GPU tensors | Part 2 FULL_TRAIN=True |
|  | Progress bars | Part 2 optional |
|  | Fuzzy matching | Part 1 (stdlib) |

---

## Notes

- Parts 1 and Bonus run with **zero pip dependencies** (stdlib only).
- Part 2 simulation mode requires only  (optional) and  (optional).
- All randomness is seeded (SEED=42) for reproducibility.
