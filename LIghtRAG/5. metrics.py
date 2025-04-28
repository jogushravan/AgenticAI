# metrics.py
from sklearn.metrics import ndcg_score
import re
import torch
from sentence_transformers import util

metrics = {
    "queries": 0,
    "successful_tasks": 0,
    "hallucination_detected": 0,
    "pii_leakage_detected": 0,
    "retries": 0,
    "reasoning_scores": [],
    "relevance_scores": [],
    "retrieval_scores": {
        "nDCG": [],
        "Recall@k": [],
        "Precision@k": [],
        "MRR": [],
        "p@1": [],
        "embedding_similarity": []
    }
}

def update_retrieval_metrics(docs, query_text):
    ranks = [1 if query_text.lower() in doc["text"].lower() else 0 for doc in docs]
    scores = [doc.get("score", 0.8) for doc in docs]
    metrics["retrieval_scores"]["nDCG"].append(ndcg_score([ranks], [scores]))
    if ranks:
        metrics["retrieval_scores"]["Recall@k"].append(sum(ranks)/len(ranks))
        metrics["retrieval_scores"]["Precision@k"].append(sum(ranks)/len(docs))
        reciprocal_ranks = [1/(i+1) for i, r in enumerate(ranks) if r == 1]
        metrics["retrieval_scores"]["MRR"].append(reciprocal_ranks[0] if reciprocal_ranks else 0)
        metrics["retrieval_scores"]["p@1"].append(1 if ranks[0] == 1 else 0)

def update_reasoning_metrics(answer, query):
    relevance_score = 1.0 if query.lower() in answer.lower() else 0.5
    metrics["reasoning_scores"].append(relevance_score)

def detect_pii(text):
    patterns = [
        r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
        r"\\b\\d{10}\\b",
        r"\\b\\d{5}(?:[-\\s]\\d{4})?\\b"
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False
