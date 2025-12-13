import os
import csv
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd


# Global objects will be initialized once and reused
_documents: List[str] = []
_embedder: SentenceTransformer | None = None
_doc_embeddings = None


def load_documents() -> List[str]:
    docs: List[str] = []

    # 1. Prioritas: data.xlsx
    xlsx_path = "data_baru.xlsx"
    if os.path.exists(xlsx_path):
        try:
            # Baca SEMUA sheet dalam data.xlsx
            all_sheets = pd.read_excel(xlsx_path, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    # Gabungkan seluruh kolom yang tidak NaN menjadi satu string
                    cells = []
                    for v in row.values:
                        if pd.isna(v):
                            continue
                        text = str(v).strip()
                        if text:
                            cells.append(text)

                    if not cells:
                        continue

                    # Sertakan nama sheet agar konteks lebih jelas
                    doc_text = f"[{sheet_name}] " + " | ".join(cells)
                    if len(doc_text) >= 20:
                        docs.append(doc_text)
        except Exception as e:
            print(f"Error membaca data.xlsx sebagai knowledge base: {e}")

    if docs:
        return docs

    # 2. Kedua: pdf_content.txt
    pdf_path = "pdf_content.txt"
    if os.path.exists(pdf_path):
        with open(pdf_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Bagi berdasarkan dua newlines (paragraf / blok)
        raw_chunks = full_text.split("\n\n")
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            # Filter chunk yang terlalu pendek
            if len(cleaned) >= 50:
                docs.append(cleaned)

    if docs:
        return docs

    # 3. Fallback ke knowledge base lama yang berbasis Q&A (ground.csv)
    kb_path = "ground.csv"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                qa_pair = f"Pertanyaan: {question}\nJawaban: {answer}"
                if len(qa_pair) > 10:
                    docs.append(qa_pair)

    return docs


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading Semantic Model (SentenceTransformer)...")
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("Semantic Model Loaded.")
    return _embedder


def get_documents_and_embeddings():
    """
    Lazy initialization untuk dokumen dan embeddingnya.
    """
    global _documents, _doc_embeddings

    if not _documents:
        _documents = load_documents()

    if _doc_embeddings is None:
        embedder = get_embedder()
        _doc_embeddings = embedder.encode(_documents, convert_to_tensor=True)

    return _documents, _doc_embeddings


def retrieve_documents(
    query: str,
    k: int = 5,
    lexical_boost: bool = True,
) -> List[Tuple[str, float]]:
    """
    Lakukan semantic search terhadap knowledge base.

    Returns list of (document_text, score) dengan skor cosine similarity (float).
    """
    documents, doc_embeddings = get_documents_and_embeddings()
    if not documents:
        return []
    embedder = get_embedder()
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    if lexical_boost:
        query_tokens = {t.lower() for t in query.split() if len(t) > 2}
        boosted_scores = []
        for i, base_score in enumerate(cosine_scores):
            text = documents[i].lower()
            bonus = 0.0
            for tok in query_tokens:
                if tok in text:
                    bonus += 0.02
            boosted_scores.append(float(base_score) + bonus)
        final_scores = torch.tensor(boosted_scores)
    else:
        final_scores = cosine_scores

    k = min(k, len(documents))
    top_scores, top_indices = torch.topk(final_scores, k=k)

    results: List[Tuple[str, float]] = []
    for score, idx in zip(top_scores, top_indices):
        results.append((documents[int(idx)], float(score)))

    return results


def build_context_from_results(
    results: List[Tuple[str, float]],
    min_score: float = 0.4,
) -> str:
    """
    Gabungkan hasil retrieve menjadi satu string konteks,
    hanya menyertakan dokumen dengan skor di atas min_score.
    """
    context_parts: List[str] = []
    for doc, score in results:
        if score >= min_score:
            context_parts.append(f"- {doc}")
    return "\n".join(context_parts)


def get_retrieval_model_objects():
    """
    Helper agar modul lain (misal evaluasi) bisa mengakses:
    - embedder
    - documents
    - doc_embeddings
    tanpa menginisialisasi ulang.
    """
    embedder = get_embedder()
    documents, doc_embeddings = get_documents_and_embeddings()
    return embedder, documents, doc_embeddings


