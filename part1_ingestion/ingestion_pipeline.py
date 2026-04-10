"""
ingestion_pipeline.py — Vexoo Labs AI Engineer Assignment Part 1
Sliding Window chunking, Knowledge Pyramid, and Hybrid Retrieval.
Author: Candidate
"""

import re, math, random, hashlib, difflib
from collections import Counter
from typing import List, Dict, Any

WINDOW_SIZE = 2500
STRIDE = 1250
TOP_K = 3
VECTOR_DIM = 10

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","is","are",
    "was","were","be","been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","this","that","these","those","it","its","we",
    "our","they","their","he","she","his","her","i","my","you","your","from","by","as",
    "not","so","if","all","more","than","can","about","also","there","when","what",
    "which","who","how","out","up","into","just","some","such","no","other","new",
    "time","only","very","even","most","over","after",
}

# ---- Sliding Window Chunking ----

def sliding_window_chunk(text: str, window: int = WINDOW_SIZE, stride: int = STRIDE) -> List[str]:
    """Split text into overlapping character-level windows."""
    chunks, start = [], 0
    while start < len(text):
        end = start + window
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += stride
    return chunks

# ---- Knowledge Pyramid Layers ----

def layer_raw_text(chunk: str) -> str:
    """Layer 0: store raw chunk as-is."""
    return chunk

def layer_chunk_summary(chunk: str) -> str:
    """Layer 1: first 2 sentences as lightweight summary."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    summary = [s for s in sentences if s][:2]
    return " ".join(summary) if summary else chunk[:200]

def layer_category(chunk: str) -> str:
    """Layer 2: rule-based domain classifier."""
    t = chunk.lower()
    if any(kw in t for kw in ["math","equation","formula","calculus","algebra","theorem","matrix","probability"]):
        return "Mathematics"
    if any(kw in t for kw in ["law","legal","court","statute","jurisdiction","contract","litigation"]):
        return "Legal"
    if any(kw in t for kw in ["health","medicine","disease","symptom","treatment","diagnosis","medical"]):
        return "Medical"
    return "General"

def _mock_tfidf_vector(chunk: str, dim: int = VECTOR_DIM) -> List[float]:
    """Deterministic mock TF-IDF vector seeded from MD5 hash of chunk."""
    digest = int(hashlib.md5(chunk.encode()).hexdigest(), 16)
    rng = random.Random(digest)
    return [round(rng.random(), 4) for _ in range(dim)]

def _extract_keywords(chunk: str, top_n: int = 5) -> List[str]:
    """Top-N most frequent non-stopword tokens."""
    tokens = [w.lower() for w in re.findall(r"[a-zA-Z]+", chunk)]
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return [w for w, _ in Counter(filtered).most_common(top_n)]

def layer_distilled_knowledge(chunk: str) -> Dict[str, Any]:
    """Layer 3: keywords + mock TF-IDF vector."""
    return {"keywords": _extract_keywords(chunk), "tfidf_vector": _mock_tfidf_vector(chunk)}

def build_pyramid(chunk: str, chunk_id: int) -> Dict[str, Any]:
    """Assemble all 4 Knowledge Pyramid layers for one chunk."""
    return {
        "chunk_id": chunk_id,
        "raw_text": layer_raw_text(chunk),
        "chunk_summary": layer_chunk_summary(chunk),
        "category": layer_category(chunk),
        "distilled_knowledge": layer_distilled_knowledge(chunk),
    }

def build_all_pyramids(chunks: List[str]) -> List[Dict[str, Any]]:
    """Build pyramids for all chunks."""
    return [build_pyramid(c, i) for i, c in enumerate(chunks)]

# ---- Retrieval ----

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na and nb else 0.0

def _fuzzy_score(q: str, s: str) -> float:
    """difflib sequence-matcher ratio between query and summary."""
    return difflib.SequenceMatcher(None, q.lower(), s.lower()).ratio()

def retrieve(query: str, pyramids: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: cosine similarity (0.6 weight) + fuzzy summary match (0.4 weight).
    Returns top-K chunks ranked by combined score.
    """
    qv = _mock_tfidf_vector(query)
    scored = []
    for p in pyramids:
        cos  = _cosine_similarity(qv, p["distilled_knowledge"]["tfidf_vector"])
        fuzz = _fuzzy_score(query, p["chunk_summary"])
        combined = 0.6*cos + 0.4*fuzz
        matched_layer = "distilled_knowledge" if cos >= fuzz else "chunk_summary"
        scored.append({
            "chunk_id": p["chunk_id"],
            "matched_layer": matched_layer,
            "cosine_score": round(cos, 4),
            "fuzzy_score": round(fuzz, 4),
            "combined_score": round(combined, 4),
            "category": p["category"],
            "snippet": p["raw_text"][:200].replace("\n", " ") + "...",
        })
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored[:top_k]

# ---- Display Helpers ----

def print_pyramid(pyr: Dict[str, Any]) -> None:
    cid = pyr["chunk_id"]
    print(f"\n{'='*60}\n  PYRAMID — Chunk #{cid}\n{'='*60}")
    print(f"[Layer 0] Raw Text (first 150 chars):\n  {pyr['raw_text'][:150].replace(chr(10),' ')}...")
    print(f"\n[Layer 1] Summary:\n  {pyr['chunk_summary']}")
    print(f"\n[Layer 2] Category: {pyr['category']}")
    print(f"\n[Layer 3] Distilled Knowledge:")
    print(f"  Keywords   : {pyr['distilled_knowledge']['keywords']}")
    print(f"  TF-IDF Vec : {pyr['distilled_knowledge']['tfidf_vector']}")

def print_results(results: List[Dict[str, Any]], query: str) -> None:
    print(f"\n{'='*60}\n  RETRIEVAL RESULTS | Query: \"{query}\"\n{'='*60}")
    for r, res in enumerate(results, 1):
        print(f"\n  Rank #{r} | Chunk {res['chunk_id']} | {res['category']} | {res['matched_layer']}")
        print(f"  Cosine={res['cosine_score']}  Fuzzy={res['fuzzy_score']}  Combined={res['combined_score']}")
        print(f"  Snippet: {res['snippet']}")

# ---- Sample Document ----

SAMPLE_DOCUMENT = """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. These processes include learning, reasoning, and self-correction.
Specific applications of AI include expert systems, natural language processing, speech
recognition, and machine vision.

Machine learning (ML) is a subset of AI that enables systems to learn and improve from
experience without being explicitly programmed. ML focuses on developing computer programs
that can access data and use it to learn for themselves. The process begins with observations
or data, such as examples, direct experience, or instruction, to look for patterns in data
and make better decisions in the future.

Deep learning is a part of a broader family of machine learning methods based on artificial
neural networks with representation learning. Learning can be supervised, semi-supervised, or
unsupervised. Deep learning architectures such as deep neural networks, recurrent neural
networks, convolutional neural networks, and transformers have been applied to fields including
computer vision, speech recognition, natural language processing, and more.

Natural language processing (NLP) is the ability of a computer program to understand human
language as it is spoken and written. NLP has been around for more than 50 years and has roots
in linguistics. NLP enables computers to understand human communication, process large amounts
of textual data, and extract insights from unstructured text.

Transformers are a deep learning model architecture introduced in the paper Attention Is All
You Need in 2017. They use a self-attention mechanism to process sequences of data, making
them highly effective for tasks like language translation, text summarisation, and question
answering. The transformer architecture is the foundation of modern large language models.

Reinforcement learning (RL) is a type of machine learning where an agent learns to make
decisions by interacting with an environment to achieve a goal. The agent receives rewards or
penalties based on the actions it takes. RL has achieved remarkable results in game playing,
robotics, and autonomous vehicle control.

Computer vision is a field of AI that enables computers to interpret and understand the visual
world. Using digital images from cameras and deep learning models, machines can accurately
identify and classify objects and then react to what they see. Applications include facial
recognition, medical image analysis, and self-driving cars.

The concept of a neural network was first introduced in 1943 by Warren McCulloch and Walter
Pitts. Modern neural networks consist of layers of interconnected nodes or neurons that
process information using connectionist approaches to computation. Each connection transmits
a signal from one neuron to another, with the output signal calculated as a non-linear
function of the sum of its inputs.

Gradient descent is an optimisation algorithm used to minimise a function by iteratively
moving in the direction of steepest descent, defined by the negative of the gradient.
In machine learning, gradient descent optimises model parameters for better predictions.
Variants include stochastic gradient descent, Adam, and RMSProp.

Overfitting occurs when a model learns the training data too well, including its noise, so
it performs poorly on new unseen data. Regularisation techniques such as L1/L2 penalties,
dropout, and early stopping are used to prevent overfitting and improve generalisation.

Attention mechanisms allow models to focus dynamically on different parts of the input when
producing output. The self-attention formula involves queries, keys, and values derived from
input embeddings. This is central to transformer architectures and captures long-range
dependencies in sequences.

Large language models such as GPT, BERT, LLaMA, and Claude are trained on massive text
corpora using self-supervised learning objectives. They learn rich language representations
and are fine-tuned for downstream tasks. Fine-tuning approaches include LoRA and prompt tuning.

AI ethics addresses concerns about fairness, accountability, transparency, and the societal
impact of AI systems. Bias in training data can lead to discriminatory outputs. Researchers
work on bias detection, interpretability, and responsible deployment to ensure AI benefits all.
"""

# ---- Main Demo ----

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   VEXOO LABS — Part 1: Document Ingestion Pipeline Demo")
    print("="*60)

    print("\n[1] Sliding window chunking...")
    chunks = sliding_window_chunk(SAMPLE_DOCUMENT)
    print(f"    Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i}: {len(c)} chars")

    print("\n[2] Building Knowledge Pyramids...")
    pyramids = build_all_pyramids(chunks)
    print(f"    Pyramids built: {len(pyramids)}")

    print("\n[3] Knowledge Pyramid — first 2 chunks:")
    for p in pyramids[:2]:
        print_pyramid(p)

    query1 = "neural networks and gradient descent optimisation"
    print(f"\n[4] Query: \"{query1}\"")
    print_results(retrieve(query1, pyramids), query1)

    query2 = "transformer attention mechanism language models"
    print(f"\n[5] Query: \"{query2}\"")
    print_results(retrieve(query2, pyramids), query2)

    print("\n" + "="*60 + "\n   Pipeline demo complete.\n" + "="*60 + "\n")
