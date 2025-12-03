# rag_utils.py
import os
import csv
from typing import List, Tuple
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import string

# --- GLOBAL OBJECTS ---
_documents: List[str] = []
_embedder: SentenceTransformer | None = None
_doc_embeddings = None
_bm25_model = None  # Untuk Sparse Retrieval
_cross_encoder = None # Untuk Reranking

def load_documents() -> List[str]:
    """
    Load knowledge base documents (Sama seperti logika sebelumnya).
    """
    docs: List[str] = []
    
    # 1. Prioritas: data.xlsx
    xlsx_path = "data.xlsx"
    if os.path.exists(xlsx_path):
        try:
            all_sheets = pd.read_excel(xlsx_path, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                if df is None or df.empty: continue
                for _, row in df.iterrows():
                    cells = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
                    if not cells: continue
                    doc_text = f"[{sheet_name}] " + " | ".join(cells)
                    if len(doc_text) >= 20: docs.append(doc_text)
        except Exception as e:
            print(f"Error reading excel: {e}")

    if docs: return docs

    # 2. Fallback: pdf_content.txt
    if os.path.exists("pdf_content.txt"):
        with open("pdf_content.txt", "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n\n")
            docs.extend([c.strip() for c in raw_chunks if len(c.strip()) >= 50])
    
    return docs

def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading Dense Model (SentenceTransformer)...")
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder

def get_cross_encoder():
    """Model untuk Reranking (Metode 3)"""
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading Cross-Encoder...")
        # Model cross-encoder yang ringan tapi efektif
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def simple_tokenize(text):
    """Tokenisasi sederhana untuk BM25"""
    text = text.lower()
    return [t for t in text.split() if t not in string.punctuation]

def initialize_retrieval_system():
    """Inisialisasi Dokumen, Embeddings (Dense), dan BM25 (Sparse)"""
    global _documents, _doc_embeddings, _bm25_model
    
    if not _documents:
        _documents = load_documents()
        
    # 1. Setup Dense (Embeddings)
    if _doc_embeddings is None:
        embedder = get_embedder()
        _doc_embeddings = embedder.encode(_documents, convert_to_tensor=True)
    
    # 2. Setup Sparse (BM25)
    if _bm25_model is None and _documents:
        tokenized_corpus = [simple_tokenize(doc) for doc in _documents]
        _bm25_model = BM25Okapi(tokenized_corpus)

    return _documents, _doc_embeddings, _bm25_model

# --- INTI FUNGSI RETRIEVAL ---

def retrieve_documents(
    query: str,
    k: int = 10,
    method: str = 'dense' # 'dense', 'hybrid', atau 'rerank'
) -> List[Tuple[str, float]]:
    
    documents, doc_embeddings, bm25_model = initialize_retrieval_system()
    if not documents: return []

    # A. METODE 1: DENSE ONLY (Cosine Similarity)
    # -------------------------------------------
    embedder = get_embedder()
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    if method == 'dense':
        top_results = torch.topk(cosine_scores, k=min(k, len(documents)))
        return [(documents[i], float(s)) for s, i in zip(top_results[0], top_results[1])]

    # B. METODE 2: HYBRID SEARCH (BM25 + Dense)
    # -----------------------------------------
    # Ambil skor BM25
    tokenized_query = simple_tokenize(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)
    
    # Normalisasi skor agar bisa dijumlahkan (Min-Max Scaling sederhana)
    def normalize(scores):
        if len(scores) == 0: return scores
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return [0.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    norm_dense = normalize(cosine_scores.tolist())
    norm_bm25 = normalize(bm25_scores)
    
    # Gabungkan: 0.5 Dense + 0.5 BM25 (Weighted Sum)
    hybrid_scores = []
    for i in range(len(documents)):
        score = (0.7 * norm_dense[i]) + (0.3 * norm_bm25[i]) # Dense biasanya lebih akurat semantic-nya
        hybrid_scores.append((score, i))
        
    hybrid_scores.sort(key=lambda x: x[0], reverse=True)
    top_hybrid = hybrid_scores[:k]
    final_results = [(documents[i], s) for s, i in top_hybrid]

    if method == 'hybrid':
        return final_results

    # C. METODE 3: CROSS-ENCODER RERANKING
    # ------------------------------------
    # Gunakan hasil Hybrid (top 2*k) sebagai kandidat, lalu di-rerank
    if method == 'rerank':
        candidates = [(documents[i], i) for s, i in hybrid_scores[:k*2]] # Ambil kandidat lebih banyak
        
        cross_encoder = get_cross_encoder()
        # Input cross encoder adalah pasangan (query, document)
        cross_inp = [[query, doc] for doc, idx in candidates]
        cross_scores = cross_encoder.predict(cross_inp)
        
        # Gabungkan skor baru dengan dokumen
        reranked_results = []
        for i in range(len(cross_scores)):
            reranked_results.append((candidates[i][0], float(cross_scores[i])))
            
        # Urutkan ulang berdasarkan skor Cross-Encoder
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:k]

    return []

def build_context_from_results(results, min_score=0.0):
    """Helper string builder"""
    return "\n".join([f"- {doc}" for doc, score in results if score >= min_score])

def get_retrieval_model_objects():
    """
    Helper khusus untuk script evaluasi agar bisa mengakses object embedder
    dan dokumen tanpa inisialisasi ulang yang berat.
    """
    # Pastikan sistem sudah terinisialisasi
    documents, doc_embeddings, _ = initialize_retrieval_system()
    embedder = get_embedder()
    
    return embedder, documents, doc_embeddings