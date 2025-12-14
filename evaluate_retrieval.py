import pandas as pd
import numpy as np
import time
import google.generativeai as genai
from sentence_transformers import util
from config import GEMINI_API_KEY

# --- IMPORT DARI RAG_UTILS_BARU (Gunakan file Anda yang sudah ada) ---
import rag_utils_baru as rag
from rag_utils_baru import (
    retrieve_documents,      # Untuk Dense & Hybrid Custom
    retrieve_bm25,           # Untuk BM25
    retrieve_hybrid_rrf,     # Untuk Hybrid RRF
    retrieve_with_rerank,    # Untuk Re-ranking
    get_embedder,
    get_documents_and_embeddings # Fungsi inisialisasi dari rag_utils
)

# [REVISI] Menggunakan file XLSX, bukan CSV
GT_PATH = "new_ground_truth.xlsx" 
OUTPUT_FILE = "hasil_evaluasi_retrieval_detail.xlsx"

# Settings
K_RETRIEVAL = 3             
RELEVANCE_THRESHOLD = 0.8  

# Setup Gemini (Optional - Hanya untuk metode HyDE & MultiQuery)
model_gen = None
if GEMINI_API_KEY and GEMINI_API_KEY != "MASUKKAN_API_KEY_ANDA":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gen = genai.GenerativeModel("gemma-2-9b-it")
    except: pass

# --- METRIC FUNCTIONS ---
def calculate_metrics(retrieved_docs, ground_truth_answer, embedder):
    """Cek relevansi menggunakan semantic similarity."""
    if not retrieved_docs or not ground_truth_answer:
        return 0.0, 0, False

    # Pastikan input string
    ground_truth_answer = str(ground_truth_answer)
    
    gt_emb = embedder.encode(ground_truth_answer, convert_to_tensor=True)
    relevant_count = 0
    is_hit = False 
    
    for doc_text, _ in retrieved_docs:
        doc_emb = embedder.encode(doc_text, convert_to_tensor=True)
        sim_score = util.cos_sim(gt_emb, doc_emb).item()
        
        if sim_score >= RELEVANCE_THRESHOLD:
            relevant_count += 1
            is_hit = True
            
    precision = relevant_count / len(retrieved_docs)
    recall_hit = 1 if is_hit else 0
    return precision, recall_hit

# --- GENERATIVE HELPERS ---
def generate_hyde(query):
    if not model_gen: return query
    try:
        res = model_gen.generate_content(f"Tuliskan jawaban singkat hipotetis untuk: {query}")
        return res.text.strip()
    except: return query

def generate_multiquery(query):
    if not model_gen: return [query]
    try:
        res = model_gen.generate_content(f"Buat 3 variasi pertanyaan dari: {query}")
        lines = [x.strip() for x in res.text.split('\n') if x.strip()]
        return lines[:3]
    except: return [query]

# --- MAIN LOOP ---
def main():
    print("\n=== EVALUASI RETRIEVAL: 8 METODE ===")
    
    # 1. SETUP KNOWLEDGE BASE
    print("[INIT] Loading Knowledge Base via rag_utils_baru...")
    # Ini akan membaca 'data_baru.xlsx' menggunakan fungsi asli di kode Anda
    rag.get_documents_and_embeddings() 
    
    if not rag._documents:
        print("ERROR: Dokumen kosong! Pastikan file 'data_baru.xlsx' ada di folder.")
        return
    else:
        print(f"[INIT] Berhasil memuat {len(rag._documents)} dokumen dari rag_utils.")

    embedder = get_embedder() 
    
    # 2. LOAD SOAL EVALUASI (GROUND TRUTH) - REVISI XLSX
    try:
        print(f"[INIT] Membaca Ground Truth dari: {GT_PATH}...")
        # [REVISI] Menggunakan read_excel
        df_gt = pd.read_excel(GT_PATH)
        
        # Deteksi kolom secara dinamis (case insensitive)
        cols = {c.lower(): c for c in df_gt.columns}
        q_col = cols.get('question') or cols.get('pertanyaan')
        a_col = cols.get('answer') or cols.get('jawaban')
        
        if not q_col or not a_col:
            print(f"Error: Kolom pertanyaan/jawaban tidak ditemukan. Kolom terdeteksi: {list(df_gt.columns)}")
            return

        questions = df_gt[q_col].dropna().tolist()
        answers = df_gt[a_col].dropna().tolist()
        
        # Limit sampel biar cepat (hapus baris ini jika ingin full)
        LIMIT = 100
        questions = questions[:LIMIT]
        answers = answers[:LIMIT]
        print(f"[INIT] Siap mengevaluasi {len(questions)} pertanyaan.")

    except Exception as e:
        print(f"FATAL: Gagal baca file GT ({GT_PATH}): {e}")
        return

    all_results = []
    
    # 3. MULAI LOOP EVALUASI
    for i, (q, gt) in enumerate(zip(questions, answers)):
        print(f"[{i+1}/{len(questions)}] Processing: {str(q)[:40]}...")
        q = str(q) # Pastikan string
        gt = str(gt)

        def run_eval(method_name, retrieved_data):
            prec, hit = calculate_metrics(retrieved_data, gt, embedder)
            
            top_doc = retrieved_data[0][0] if retrieved_data else "NO RESULT"
            top_score = retrieved_data[0][1] if retrieved_data else 0.0
            
            all_results.append({
                "Method": method_name,
                "Question": q,
                "Ground Truth": gt,
                "Top-1 Retrieved": top_doc[:300], 
                "Score": top_score,
                "Precision": prec,
                "Recall (Hit)": hit
            })

        # --- JALANKAN 8 METODE ---

        # 1. Naive Lexical (Manual)
        kb_docs = rag._documents 
        q_set = set(q.lower().split())
        lex_scores = []
        for d in kb_docs:
            d_set = set(d.lower().split())
            sc = len(q_set & d_set) / len(q_set) if q_set else 0
            lex_scores.append((d, sc))
        run_eval("1. Naive Lexical", sorted(lex_scores, key=lambda x:x[1], reverse=True)[:K_RETRIEVAL])

        # 2. Dense (SBERT)
        run_eval("2. Dense", retrieve_documents(q, k=K_RETRIEVAL, lexical_boost=False))

        # 3. Hybrid Custom
        run_eval("3. Hybrid Custom", retrieve_documents(q, k=K_RETRIEVAL, lexical_boost=True))

        # 4. BM25
        run_eval("4. BM25", retrieve_bm25(q, k=K_RETRIEVAL))

        # 5. Hybrid RRF
        run_eval("5. Hybrid RRF", retrieve_hybrid_rrf(q, k=K_RETRIEVAL))

        # 6. Re-Ranking
        run_eval("6. Re-Ranking", retrieve_with_rerank(q, k=K_RETRIEVAL))

        # 7. HyDE (Perlu API Key)
        if model_gen:
            hyde_q = generate_hyde(q)
            run_eval("7. HyDE", retrieve_documents(hyde_q, k=K_RETRIEVAL, lexical_boost=False))
        else:
            all_results.append({"Method": "7. HyDE", "Question": q, "Top-1 Retrieved": "SKIPPED"})

        # 8. Multi-Query (Perlu API Key)
        if model_gen:
            qs = generate_multiquery(q)
            combined = []
            for sub_q in qs: combined.extend(retrieve_documents(sub_q, k=2, lexical_boost=False))
            # Deduplikasi
            seen, unique = set(), []
            for d, s in combined:
                if d not in seen:
                    unique.append((d,s))
                    seen.add(d)
            unique.sort(key=lambda x:x[1], reverse=True)
            run_eval("8. Multi-Query", unique[:K_RETRIEVAL])
        else:
            all_results.append({"Method": "8. Multi-Query", "Question": q, "Top-1 Retrieved": "SKIPPED"})

    # 4. SAVE
    df_res = pd.DataFrame(all_results)
    print("\n=== SUMMARY SCORES ===")
    print(df_res.groupby("Method")[["Precision", "Recall (Hit)"]].mean())
    
    df_res.to_excel(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()