import os
import csv
from typing import List, Tuple, Dict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pandas as pd
from rank_bm25 import BM25Okapi

# Global objects
_documents: List[str] = []
_embedder: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None
_doc_embeddings = None
_bm25_model: BM25Okapi | None = None

def load_documents() -> List[str]:
    docs: List[str] = []
    xlsx_path = "data_baru.xlsx"
    if os.path.exists(xlsx_path):
        try:
            all_sheets = pd.read_excel(xlsx_path, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                if df is None or df.empty: continue
                for _, row in df.iterrows():
                    cells = [str(v).strip() for v in row.values if not pd.isna(v) and str(v).strip()]
                    if not cells: continue
                    doc_text = f"[{sheet_name}] " + " | ".join(cells)
                    if len(doc_text) >= 20: docs.append(doc_text)
        except Exception as e: print(f"Error reading excel: {e}")

    if docs: return docs

    # 2. Kedua: pdf_content.txt
    pdf_path = "pdf_content.txt"
    if os.path.exists(pdf_path):
        with open(pdf_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n\n")
            for chunk in raw_chunks:
                if len(chunk.strip()) >= 50: docs.append(chunk.strip())

    if docs: return docs

    # 3. Fallback: ground.csv
    kb_path = "ground.csv"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qa = f"Pertanyaan: {row.get('question','')}\nJawaban: {row.get('answer','')}"
                if len(qa) > 10: docs.append(qa)
    return docs

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading Bi-Encoder (SentenceTransformer)...")
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder

def get_cross_encoder() -> CrossEncoder:
    """Load Cross-Encoder model for Re-Ranking"""
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading Cross-Encoder...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def get_documents_and_embeddings():
    global _documents, _doc_embeddings, _bm25_model

    if not _documents:
        _documents = load_documents()

    # Init Dense Embeddings
    if _doc_embeddings is None:
        embedder = get_embedder()
        _doc_embeddings = embedder.encode(_documents, convert_to_tensor=True)
    
    # Init BM25 Index (New)
    if _bm25_model is None:
        print("Building BM25 Index...")
        tokenized_corpus = [doc.lower().split() for doc in _documents]
        _bm25_model = BM25Okapi(tokenized_corpus)

    return _documents, _doc_embeddings

def retrieve_documents(query: str, k: int = 5, lexical_boost: bool = True) -> List[Tuple[str, float]]:
    """Original Dense & Custom Hybrid Logic"""
    documents, doc_embeddings = get_documents_and_embeddings()
    if not documents: return []
    embedder = get_embedder()
    
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    if lexical_boost: # Method 3: Custom Hybrid
        query_tokens = {t.lower() for t in query.split() if len(t) > 2}
        boosted_scores = []
        for i, base_score in enumerate(cosine_scores):
            text = documents[i].lower()
            bonus = sum(0.02 for tok in query_tokens if tok in text)
            boosted_scores.append(float(base_score) + bonus)
        final_scores = torch.tensor(boosted_scores)
    else: # Method 2: Pure Dense
        final_scores = cosine_scores

    top_scores, top_indices = torch.topk(final_scores, k=min(k, len(documents)))
    return [(documents[int(idx)], float(score)) for score, idx in zip(top_scores, top_indices)]

def retrieve_bm25(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Method 4: Pure BM25"""
    documents, _ = get_documents_and_embeddings() # Ensure loaded
    global _bm25_model
    
    tokenized_query = query.lower().split()
    # Get top-k documents
    top_docs = _bm25_model.get_top_n(tokenized_query, documents, n=k)
    # BM25 library doesn't return scores easily in top_n, so we recalculate for those
    results = []
    doc_scores = _bm25_model.get_scores(tokenized_query)
    # Map back to get scores
    top_indices = np.argsort(doc_scores)[::-1][:k]
    for idx in top_indices:
        results.append((documents[idx], float(doc_scores[idx])))
    return results

def retrieve_hybrid_rrf(query: str, k: int = 5, k_rrf: int = 60) -> List[Tuple[str, float]]:
    """Method 5: Hybrid using Reciprocal Rank Fusion (BM25 + Dense)"""
    documents, doc_embeddings = get_documents_and_embeddings()
    embedder = get_embedder()
    global _bm25_model

    # 1. Get Dense Results (Indices)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    dense_top_k = torch.topk(dense_scores, k=min(len(documents), k * 2)).indices.tolist()

    # 2. Get BM25 Results (Indices)
    tokenized_query = query.lower().split()
    bm25_scores = _bm25_model.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[::-1][:min(len(documents), k * 2)]

    # 3. Fuse Ranks (RRF Algorithm)
    rrf_score = {}
    for rank, idx in enumerate(dense_top_k):
        rrf_score[idx] = rrf_score.get(idx, 0) + (1 / (k_rrf + rank + 1))
    for rank, idx in enumerate(bm25_top_k):
        rrf_score[idx] = rrf_score.get(idx, 0) + (1 / (k_rrf + rank + 1))

    # Sort by RRF score
    sorted_rrf = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return [(documents[idx], score) for idx, score in sorted_rrf]

def retrieve_with_rerank(query: str, k: int = 5, fetch_k: int = 20) -> List[Tuple[str, float]]:
    documents, _ = get_documents_and_embeddings()
    # 1. First Stage: Retrieve more candidates (fetch_k) using fast Dense
    candidates = retrieve_documents(query, k=fetch_k, lexical_boost=False) # Get simple dense results
    
    if not candidates: return []

    # 2. Second Stage: Re-Rank with Cross-Encoder
    ce = get_cross_encoder()
    ce_inputs = [[query, doc_text] for doc_text, _ in candidates]
    ce_scores = ce.predict(ce_inputs)

    # Combine and Sort
    reranked = list(zip(candidates, ce_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    # Return top k format: (doc_text, new_score)
    final_results = [(item[0][0], float(item[1])) for item in reranked[:k]]
    return final_results

def build_context_from_results(results: List[Tuple[str, float]], min_score: float = 0.4) -> str:
    context_parts = []
    for doc, score in results:
        # Untuk BM25/RRF skornya bisa bervariasi, jadi min_score mungkin perlu diabaikan atau disesuaikan
        # Kita set constraint sederhana agar tidak menyaring terlalu agresif
        if score > -999: 
            context_parts.append(f"- {doc}")
    return "\n".join(context_parts)
