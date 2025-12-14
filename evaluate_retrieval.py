import pandas as pd
import numpy as np
import time
import torch
import google.generativeai as genai
from sentence_transformers import util
from config import GEMINI_API_KEY
from rank_bm25 import BM25Okapi
import rag_utils_baru as rag
from rag_utils_baru import (
    retrieve_documents,
    retrieve_bm25,
    retrieve_hybrid_rrf,
    retrieve_with_rerank,
    get_embedder,
    get_documents_and_embeddings
)

GT_PATH = "new_ground_truth_baru.xlsx" 
OUTPUT_FILE = "hasil_evaluasi_retrieval_detail.xlsx"

K_VALUES = [1, 3, 5, 10, 20]
RELEVANCE_THRESHOLD = 0.6 

model_gen = None
if GEMINI_API_KEY and GEMINI_API_KEY != "MASUKKAN_API_KEY_ANDA":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gen = genai.GenerativeModel("gemma-3-27b-it")
        print("[INIT] Gemini model loaded: gemma-3-27b-it")
    except Exception as e:
        print(f"[WARNING] Gagal load Gemini model: {e}")

def calculate_metrics(retrieved_docs, ground_truth_answer, embedder):
    """Cek relevansi menggunakan semantic similarity."""
    if not retrieved_docs or not ground_truth_answer:
        return 0.0, 0

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
            
    precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0.0
    recall_hit = 1 if is_hit else 0
    return precision, recall_hit

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

def main():
    print("\n=== EVALUASI RETRIEVAL: 8 METODE ===")
    
    rag._documents = []
    rag._doc_embeddings = None
    rag._bm25_model = None
    
    try:
        print(f"[INIT] Membaca Ground Truth dari: {GT_PATH}...")
        df_gt = pd.read_excel(GT_PATH)
        
        print(f"[DEBUG] Kolom yang ditemukan: {list(df_gt.columns)}")
        
        cols = {c.lower(): c for c in df_gt.columns}
        q_col = cols.get('question') or cols.get('pertanyaan') or cols.get('query')
        a_col = cols.get('answer') or cols.get('jawaban') or cols.get('response')
        
        source_id_col = cols.get('source_id')
        source_title_col = cols.get('source_title')
        source_section_col = cols.get('source_section')
        source_type_col = cols.get('source_type')
        
        if not q_col or not a_col:
            print(f"Error: Kolom pertanyaan/jawaban tidak ditemukan.")
            print(f"Kolom yang tersedia: {list(df_gt.columns)}")
            print(f"Mencoba menggunakan kolom pertama dan kedua...")
            if len(df_gt.columns) >= 2:
                q_col = df_gt.columns[0]
                a_col = df_gt.columns[1]
            else:
                return

        questions = df_gt[q_col].dropna().tolist()
        answers = df_gt[a_col].dropna().tolist()
        
        if source_id_col:
            source_ids = df_gt[source_id_col].dropna().tolist()
            print(f"[INFO] Ground truth memiliki {len([s for s in source_ids if pd.notna(s)])} referensi source_id")
        
        LIMIT = 100
        questions = questions[:LIMIT]
        answers = answers[:LIMIT]
        print(f"[INIT] Siap mengevaluasi {len(questions)} pertanyaan.")
        print(f"[INFO] Tujuan evaluasi: Mengukur apakah retrieval berhasil menemukan dokumen relevan")

    except Exception as e:
        print(f"FATAL: Gagal baca file GT ({GT_PATH}): {e}")
        import traceback
        traceback.print_exc()
        return

    print("[INIT] Loading Knowledge Base dari data_baru.xlsx...")
    embedder = get_embedder()
    
    get_documents_and_embeddings(exclude_ground_truth=False)
    
    if not rag._documents:
        print("ERROR: Dokumen kosong! Pastikan file 'data_baru.xlsx' ada di folder.")
        return
    else:
        print(f"[INIT] Berhasil memuat {len(rag._documents)} dokumen dari knowledge base.")

    all_results = []
    
    # 4. MULAI LOOP EVALUASI
    print(f"\n[EVAL] Memulai evaluasi dengan k values: {K_VALUES}")
    print(f"[EVAL] Total pertanyaan: {len(questions)}")
    print(f"[EVAL] Total metode: 8")
    print(f"[EVAL] Total kombinasi: {len(questions)} x 8 x {len(K_VALUES)} = {len(questions) * 8 * len(K_VALUES)}\n")
    
    for i, (q, gt) in enumerate(zip(questions, answers)):
        q = str(q)
        gt = str(gt)
        
        print(f"\n[{i+1}/{len(questions)}] Question: {q[:60]}...")
        
        def save_result(method_name, k_value, retrieved_data, precision, recall_hit):
            """Simpan hasil evaluasi untuk satu metode dan satu nilai k"""
            top_doc = retrieved_data[0][0] if retrieved_data else "NO RESULT"
            top_score = retrieved_data[0][1] if retrieved_data else 0.0
            
            all_results.append({
                "Method": method_name,
                "K": k_value,
                "Question": q,
                "Ground Truth": gt,
                "Top-1 Retrieved": top_doc[:300], 
                "Top Score": top_score,
                "Precision": precision,
                "Recall (Hit)": recall_hit
            })

        kb_docs = rag._documents 
        q_set = set(q.lower().split())
        lex_scores = []
        for d in kb_docs:
            d_set = set(d.lower().split())
            sc = len(q_set & d_set) / len(q_set) if q_set else 0
            lex_scores.append((d, sc))
        lex_scores_sorted = sorted(lex_scores, key=lambda x:x[1], reverse=True)
        
        for k in K_VALUES:
            retrieved = lex_scores_sorted[:k]
            prec, hit = calculate_metrics(retrieved, gt, embedder)
            save_result("1. Naive Lexical", k, retrieved, prec, hit)

        for k in K_VALUES:
            retrieved = retrieve_documents(q, k=k, lexical_boost=False)
            prec, hit = calculate_metrics(retrieved, gt, embedder)
            save_result("2. Dense", k, retrieved, prec, hit)

        for k in K_VALUES:
            retrieved = retrieve_documents(q, k=k, lexical_boost=True)
            prec, hit = calculate_metrics(retrieved, gt, embedder)
            save_result("3. Hybrid Custom", k, retrieved, prec, hit)

        for k in K_VALUES:
            retrieved = retrieve_bm25(q, k=k)
            prec, hit = calculate_metrics(retrieved, gt, embedder)
            save_result("4. BM25", k, retrieved, prec, hit)

        for k in K_VALUES:
            retrieved = retrieve_hybrid_rrf(q, k=k)
            prec, hit = calculate_metrics(retrieved, gt, embedder)
            save_result("5. Hybrid RRF", k, retrieved, prec, hit)

        for k in K_VALUES:
            fetch_k = max(k * 2, 20)
            retrieved = retrieve_with_rerank(q, k=k, fetch_k=fetch_k)
            prec, hit = calculate_metrics(retrieved, gt, embedder)
            save_result("6. Re-Ranking", k, retrieved, prec, hit)

        if model_gen:
            hyde_q = generate_hyde(q)
            for k in K_VALUES:
                retrieved = retrieve_documents(hyde_q, k=k, lexical_boost=False)
                prec, hit = calculate_metrics(retrieved, gt, embedder)
                save_result("7. HyDE", k, retrieved, prec, hit)
        else:
            for k in K_VALUES:
                all_results.append({
                    "Method": "7. HyDE",
                    "K": k,
                    "Question": q,
                    "Ground Truth": gt,
                    "Top-1 Retrieved": "SKIPPED (No API Key)",
                    "Top Score": 0.0,
                    "Precision": 0.0,
                    "Recall (Hit)": 0
                })

        if model_gen:
            qs = generate_multiquery(q)
            combined = []
            for sub_q in qs:
                combined.extend(retrieve_documents(sub_q, k=max(K_VALUES), lexical_boost=False))
            seen, unique = set(), []
            for d, s in combined:
                if d not in seen:
                    unique.append((d, s))
                    seen.add(d)
            unique.sort(key=lambda x:x[1], reverse=True)
            
            for k in K_VALUES:
                retrieved = unique[:k]
                prec, hit = calculate_metrics(retrieved, gt, embedder)
                save_result("8. Multi-Query", k, retrieved, prec, hit)
        else:
            for k in K_VALUES:
                all_results.append({
                    "Method": "8. Multi-Query",
                    "K": k,
                    "Question": q,
                    "Ground Truth": gt,
                    "Top-1 Retrieved": "SKIPPED (No API Key)",
                    "Top Score": 0.0,
                    "Precision": 0.0,
                    "Recall (Hit)": 0
                })
        
        if (i + 1) % 10 == 0:
            print(f"[PROGRESS] Selesai {i+1}/{len(questions)} pertanyaan...")

    print("\n" + "="*80)
    print("ANALISIS HASIL EVALUASI")
    print("="*80)
    
    df_res = pd.DataFrame(all_results)
    
    print("\n--- AVERAGE PRECISION PER METODE DAN PER K ---")
    summary_k = df_res.groupby(["Method", "K"])[["Precision", "Recall (Hit)"]].mean().reset_index()
    summary_k = summary_k.pivot(index="Method", columns="K", values="Precision")
    print(summary_k.round(4))
    
    print("\n--- AVERAGE PRECISION PER METODE (Rata-rata semua K) ---")
    summary_method = df_res.groupby("Method")[["Precision", "Recall (Hit)"]].mean().reset_index()
    summary_method = summary_method.sort_values("Precision", ascending=False)
    print(summary_method.round(4))
    
    print("\n--- AVERAGE PRECISION PER K (Rata-rata semua Metode) ---")
    summary_k_avg = df_res.groupby("K")[["Precision", "Recall (Hit)"]].mean().reset_index()
    print(summary_k_avg.round(4))
    
    print(f"\n[SAVE] Menyimpan hasil detail ke: {OUTPUT_FILE}")
    df_res.to_excel(OUTPUT_FILE, index=False)
    
    summary_file = OUTPUT_FILE.replace(".xlsx", "_summary.xlsx")
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        summary_method.to_excel(writer, sheet_name="Avg Per Method", index=False)
        summary_k_avg.to_excel(writer, sheet_name="Avg Per K", index=False)
        summary_k.to_excel(writer, sheet_name="Precision Matrix")
    
    print(f"[SAVE] Menyimpan summary ke: {summary_file}")
    print("\n" + "="*80)
    print("EVALUASI SELESAI!")
    print("="*80)

if __name__ == "__main__":
    main()