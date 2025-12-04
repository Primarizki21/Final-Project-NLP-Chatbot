import os
import pandas as pd
import numpy as np
import torch
import string
from typing import List, Tuple

# Libraries untuk retrieval methods
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi

# --- GLOBAL OBJECTS (Caching agar tidak load berulang kali) ---
_documents: List[str] = []
_embedder: SentenceTransformer | None = None  # Untuk Dense & Hybrid
_doc_embeddings = None                       # Vektor Dense Dokumen
_bm25_model = None                           # Untuk Hybrid (Sparse)
_cross_encoder = None                        # Untuk Rerank
_tfidf_vectorizer = None                     # Untuk TF-IDF
_tfidf_matrix = None                         # Matrix TF-IDF Dokumen
_doc2vec_model = None                        # Untuk Doc2Vec

def load_documents() -> List[str]:
    """Load documents dari Excel atau TXT."""
    docs: List[str] = []
    
    # 1. Coba load dari Excel
    xlsx_path = "data.xlsx"
    if os.path.exists(xlsx_path):
        try:
            all_sheets = pd.read_excel(xlsx_path, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                if df is None or df.empty: continue
                # Gabungkan semua kolom menjadi satu teks per baris
                for _, row in df.iterrows():
                    cells = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
                    if not cells: continue
                    doc_text = f"[{sheet_name}] " + " | ".join(cells)
                    if len(doc_text) >= 10: docs.append(doc_text)
        except Exception as e:
            print(f"Error reading excel: {e}")

    # 2. Jika Excel kosong/gagal, coba Text file
    if not docs and os.path.exists("pdf_content.txt"):
        with open("pdf_content.txt", "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n\n")
            docs.extend([c.strip() for c in raw_chunks if len(c.strip()) >= 20])
    
    return docs

def simple_tokenize(text: str) -> List[str]:
    """Tokenisasi sederhana: lowercase + remove punctuation."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

def initialize_retrieval_system():
    """Inisialisasi/Training model retrieval (Dense, BM25, TF-IDF, Doc2Vec) sekali jalan."""
    global _documents, _doc_embeddings, _embedder, _bm25_model
    global _tfidf_vectorizer, _tfidf_matrix, _doc2vec_model
    
    # 1. Load Dokumen
    if not _documents:
        _documents = load_documents()
        if not _documents:
            print("Warning: Tidak ada dokumen yang dimuat.")
            return
        
    # 2. Setup Dense (SBERT)
    if _doc_embeddings is None:
        print("Initializing Dense Retrieval (SBERT)...")
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _doc_embeddings = _embedder.encode(_documents, convert_to_tensor=True)
    
    # 3. Setup BM25
    if _bm25_model is None:
        print("Initializing BM25...")
        tokenized_corpus = [simple_tokenize(doc) for doc in _documents]
        _bm25_model = BM25Okapi(tokenized_corpus)

    # 4. Setup TF-IDF
    if _tfidf_vectorizer is None:
        print("Initializing TF-IDF...")
        _tfidf_vectorizer = TfidfVectorizer()
        _tfidf_matrix = _tfidf_vectorizer.fit_transform(_documents)

    # 5. Setup Doc2Vec (Training on the fly)
    if _doc2vec_model is None:
        print("Training Doc2Vec Model...")
        tagged_data = [TaggedDocument(words=simple_tokenize(doc), tags=[i]) 
                       for i, doc in enumerate(_documents)]
        # Parameter: vector_size=50, epochs=40 (cukup cepat untuk dataset kecil-menengah)
        model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        _doc2vec_model = model

    return _documents

def retrieve_documents(query: str, k: int = 5, method: str = 'dense') -> List[Tuple[str, float]]:
    """
    Fungsi utama untuk mengambil dokumen relevan berdasarkan metode yang dipilih.
    Method options: 'dense', 'hybrid', 'rerank', 'jaccard', 'tfidf', 'doc2vec'
    """
    initialize_retrieval_system()
    if not _documents: return []
    
    results = []

    # --- 1. DENSE RETRIEVAL (Cosine Sim) ---
    if method == 'dense':
        query_embedding = _embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, _doc_embeddings)[0]
        top_results = torch.topk(scores, k=min(k, len(_documents)))
        results = [(_documents[i], float(s)) for s, i in zip(top_results[0], top_results[1])]

    # --- 2. HYBRID (BM25 + Dense) ---
    elif method == 'hybrid':
        # Dense Score
        q_emb = _embedder.encode(query, convert_to_tensor=True)
        dense_scores = util.cos_sim(q_emb, _doc_embeddings)[0].cpu().numpy()
        
        # BM25 Score
        bm25_scores = np.array(_bm25_model.get_scores(simple_tokenize(query)))
        
        # Normalisasi (Min-Max) agar skala sama (0-1)
        def normalize(arr):
            if arr.max() == arr.min(): return arr
            return (arr - arr.min()) / (arr.max() - arr.min())
            
        norm_dense = normalize(dense_scores)
        norm_bm25 = normalize(bm25_scores)
        
        # Gabung (Bobot 0.5 masing-masing)
        combined_scores = 0.5 * norm_dense + 0.5 * norm_bm25
        top_indices = np.argsort(combined_scores)[::-1][:k]
        results = [(_documents[i], float(combined_scores[i])) for i in top_indices]

    # --- 3. RERANK (Retrieve Dense -> Rerank CrossEncoder) ---
    elif method == 'rerank':
        # Retrieve Candidate (Top-20 via Dense)
        candidates_res = retrieve_documents(query, k=20, method='dense')
        candidates = [res[0] for res in candidates_res]
        
        global _cross_encoder
        if _cross_encoder is None:
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
        cross_inp = [[query, doc] for doc in candidates]
        cross_scores = _cross_encoder.predict(cross_inp)
        
        # Sort berdasarkan cross_score
        reranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)[:k]
        results = [(doc, float(score)) for doc, score in reranked]

    # --- 4. JACCARD SIMILARITY ---
    elif method == 'jaccard':
        q_tokens = set(simple_tokenize(query))
        scores = []
        for i, doc in enumerate(_documents):
            d_tokens = set(simple_tokenize(doc))
            if not q_tokens or not d_tokens:
                score = 0.0
            else:
                intersection = len(q_tokens.intersection(d_tokens))
                union = len(q_tokens.union(d_tokens))
                score = intersection / union
            scores.append((score, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        # Ambil top k
        top_k_scores = scores[:k]
        results = [(_documents[i], s) for s, i in top_k_scores]

    # --- 5. TF-IDF SUM ---
    elif method == 'tfidf':
        query_vec = _tfidf_vectorizer.transform([query])
        # Hitung cosine similarity antara vector query dan seluruh dokumen
        cosine_sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_sims)[::-1][:k]
        results = [(_documents[i], float(cosine_sims[i])) for i in top_indices]

    # --- 6. DOC2VEC ---
    elif method == 'doc2vec':
        q_tokens = simple_tokenize(query)
        # Infer vector dari query
        q_vec = _doc2vec_model.infer_vector(q_tokens)
        # Cari dokumen similar
        sims = _doc2vec_model.dv.most_similar([q_vec], topn=k)
        # sims return list of (tag, score), tag kita simpan sebagai index int tadi
        results = [(_documents[int(tag)], float(score)) for tag, score in sims]

    return results

def build_context_from_results(results: List[Tuple[str, float]], min_score=-100.0):
    """Menggabungkan hasil retrieval menjadi satu string konteks."""
    # Filter score jika perlu, tapi default -100 ambil semua hasil retrieval
    valid_docs = [doc for doc, score in results if score >= min_score]
    return "\n".join([f"- {doc}" for doc in valid_docs])

def get_retrieval_model_objects():
    """Helper untuk evaluasi script mengakses object embedder."""
    initialize_retrieval_system()
    return _embedder, _documents, _doc_embeddings