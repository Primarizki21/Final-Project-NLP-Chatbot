"""
Skrip Evaluasi Chatbot Lengkap (Ablation Study)

Fitur:
1. Membandingkan 3 Metode Retrieval: Dense, Hybrid (BM25+Dense), Rerank.
2. Membandingkan 3 Model Generation: Gemini 2.0 Flash, Cohere Command R, Llama 3.3.

Cara menjalankan:
    python evaluate_chatbot.py
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Tuple
from rouge_score import rouge_scorer
from sentence_transformers import util
from config import GEMINI_API_KEY, COHERE_API_KEY, GROQ_API_KEY

# Import library API
import google.generativeai as genai
import cohere
from groq import Groq

# Import dari modul rag_utils yang baru
from rag_utils import (
    retrieve_documents,
    get_retrieval_model_objects, # Pastikan fungsi ini ada di rag_utils (sesuai update sebelumnya)
    build_context_from_results,
    initialize_retrieval_system
)

# --- KONFIGURASI API KEY ---
# Ambil dari Environment Variable atau set manual string kosong jika tidak ada
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- SETUP CLIENTS (Lazy Loading) ---
co_client = None
groq_client = None

if COHERE_API_KEY:
    co_client = cohere.Client(COHERE_API_KEY)
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def load_ground_truth_from_excel(path: str) -> tuple[list[str], list[str]]:
    """Memuat pertanyaan dan jawaban ground truth dari Excel."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan.")

    df = pd.read_excel(path)
    
    # Deteksi nama kolom fleksibel
    cols = df.columns.str.lower()
    q_col = next((c for c in df.columns if 'question' in c.lower() or 'tanya' in c.lower()), None)
    a_col = next((c for c in df.columns if 'answer' in c.lower() or 'jawab' in c.lower()), None)

    if not q_col or not a_col:
        raise ValueError("Kolom 'question' dan 'answer' tidak ditemukan di Excel.")

    questions = [str(v).strip() for v in df[q_col] if str(v).strip()]
    answers = [str(v).strip() for v in df[a_col] if str(v).strip()]
    
    limit = min(len(questions), len(answers))
    return questions[:limit], answers[:limit]


def evaluate_retrieval_methods(questions: List[str], answers: List[str]):
    """
    Membandingkan performa Dense, Hybrid, dan Rerank.
    """
    print("\n" + "="*50)
    print("MULAI EVALUASI RETRIEVAL (Dense vs Hybrid vs Rerank)")
    print("="*50)

    # Inisialisasi sistem sekali saja
    documents, doc_embeddings, _ = initialize_retrieval_system()
    embedder = get_retrieval_model_objects()[0] # Ambil object embedder

    if not documents:
        print("Knowledge base kosong.")
        return

    # Pre-compute embedding jawaban ground truth untuk menentukan relevansi
    print("Menghitung embedding ground truth...")
    gt_embeddings = embedder.encode(answers, convert_to_tensor=True)
    
    methods = ['dense', 'hybrid', 'rerank']
    results_summary = []

    # Loop untuk setiap metode retrieval
    for method in methods:
        print(f"\nTesting Metode: {method.upper()} ...")
        
        precisions = []
        recalls = []
        
        # Parameter retrieval
        k_retrieve = 5 
        
        # Threshold untuk menentukan apakah dokumen di DB dianggap "Relevan" dengan Ground Truth Answer
        # (Ini adalah 'Gold Standard' kita)
        relevance_threshold = 0.5 

        for i, (q, gt_emb) in enumerate(zip(questions, gt_embeddings)):
            # 1. Tentukan "Kunci Jawaban" (Dokumen mana di DB yang relevan dengan GT Answer?)
            #    Kita pakai Cosine Similarity murni di sini sebagai oracle.
            sim_scores = util.cos_sim(gt_emb, doc_embeddings)[0]
            relevant_doc_indices = [
                idx for idx, score in enumerate(sim_scores) 
                if float(score) >= relevance_threshold
            ]

            # Jika tidak ada dokumen yang relevan di DB (mungkin info baru), skip hitungan
            if not relevant_doc_indices:
                continue

            # 2. Lakukan Retrieval menggunakan metode yang sedang dites
            #    Hasilnya adalah list of (text, score)
            retrieved_results = retrieve_documents(q, k=k_retrieve, method=method)
            
            # 3. Hitung intersection (irisan)
            retrieved_hits = 0
            retrieved_indices = []
            
            # Mapping teks balik ke index dokumen asli untuk pencocokan
            for r_text, _ in retrieved_results:
                try:
                    # Cari index dokumen tersebut di list utama
                    # Note: ini linear search sederhana, bisa dioptimasi tapi cukup untuk skripsi
                    idx = documents.index(r_text)
                    retrieved_indices.append(idx)
                except ValueError:
                    continue
            
            # Hitung berapa dokumen yang terambil yang BENAR-BENAR relevan
            retrieved_hits = len(set(retrieved_indices).intersection(set(relevant_doc_indices)))
            
            # Hitung Precision & Recall
            prec = retrieved_hits / k_retrieve if k_retrieve > 0 else 0
            rec = retrieved_hits / len(relevant_doc_indices) if len(relevant_doc_indices) > 0 else 0
            
            precisions.append(prec)
            recalls.append(rec)

        # Rata-rata untuk metode ini
        avg_p = sum(precisions) / len(precisions) if precisions else 0
        avg_r = sum(recalls) / len(recalls) if recalls else 0
        
        print(f"  -> Precision: {avg_p:.4f}")
        print(f"  -> Recall   : {avg_r:.4f}")
        
        results_summary.append({
            "Method": method,
            "Avg Precision": avg_p,
            "Avg Recall": avg_r
        })

    # Tampilkan Tabel Akhir
    print("\n--- HASIL AKHIR RETRIEVAL ---")
    df_res = pd.DataFrame(results_summary)
    print(df_res)
    return df_res


def generate_with_model(model_name, prompt):
    """Wrapper untuk memanggil berbagai API model."""
    try:
        if model_name == 'gemini':
            if not GEMINI_API_KEY: return "Error: No API Key"
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content(prompt).text.strip()
            
        elif model_name == 'cohere':
            if not co_client: return "Error: No API Key"
            response = co_client.chat(message=prompt, model='command-r', temperature=0.3)
            return response.text.strip()
            
        elif model_name == 'llama':
            if not groq_client: return "Error: No API Key"
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            return completion.choices[0].message.content.strip()
            
    except Exception as e:
        return f"Error Generating: {str(e)}"
    return ""


def evaluate_generation_models(questions: List[str], answers: List[str]):
    """
    Membandingkan ROUGE score dari Gemini, Cohere, dan Llama.
    Menggunakan metode retrieval TERBAIK (misal: 'rerank') sebagai basis.
    """
    print("\n" + "="*50)
    print("MULAI EVALUASI GENERATION (ROUGE SCORE)")
    print("="*50)
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    models_to_test = []
    
    # Cek ketersediaan kunci
    if GEMINI_API_KEY: models_to_test.append('gemini')
    if COHERE_API_KEY: models_to_test.append('cohere')
    if GROQ_API_KEY: models_to_test.append('llama')
    
    if not models_to_test:
        print("Tidak ada API Key generation yang ditemukan. Skip evaluasi generation.")
        return

    # Gunakan metode retrieval terbaik (misal 'rerank') untuk feeding konteks
    best_retrieval_method = 'rerank' 
    
    summary_rouge = []
    
    # Batasi sampel biar tidak tekor kuota API (misal 10-20 sampel saja untuk tes)
    # Set ke len(questions) untuk evaluasi penuh
    MAX_SAMPLES = min(20, len(questions)) 
    print(f"Evaluasi dilakukan pada {MAX_SAMPLES} sampel pertanyaan.")

    for model_name in models_to_test:
        print(f"\nTesting Model: {model_name.upper()} ...")
        
        r1_scores = []
        rl_scores = []
        
        for i in range(MAX_SAMPLES):
            q = questions[i]
            gt_ans = answers[i]
            
            # 1. Retrieve Context
            results = retrieve_documents(q, k=5, method=best_retrieval_method)
            context = build_context_from_results(results, min_score=-100)
            
            # 2. Construct Prompt (Sama rata untuk semua model agar adil)
            prompt = f"""
            Anda adalah asisten dosen wali. Jawab pertanyaan mahasiswa berikut berdasarkan konteks yang diberikan.
            
            Konteks:
            {context}
            
            Pertanyaan: {q}
            
            Jawaban (Bahasa Indonesia):
            """
            
            # 3. Generate
            pred_ans = generate_with_model(model_name, prompt)
            
            # 4. Calculate ROUGE
            if "Error" not in pred_ans:
                scores = scorer.score(gt_ans, pred_ans)
                r1_scores.append(scores["rouge1"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)
                
                # Print progress kecil-kecilan
                if i % 5 == 0:
                    print(f"  Sampel {i+1}/{MAX_SAMPLES} selesai.")
            else:
                print(f"  Gagal generate sampel {i}: {pred_ans}")

        # Rata-rata
        avg_r1 = sum(r1_scores) / len(r1_scores) if r1_scores else 0
        avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0
        
        print(f"  -> ROUGE-1: {avg_r1:.4f}")
        print(f"  -> ROUGE-L: {avg_rl:.4f}")
        
        summary_rouge.append({
            "Model": model_name,
            "Avg ROUGE-1": avg_r1,
            "Avg ROUGE-L": avg_rl
        })

    print("\n--- HASIL AKHIR GENERATION ---")
    df_gen = pd.DataFrame(summary_rouge)
    print(df_gen)


if __name__ == "__main__":
    # Pastikan file excel tersedia
    file_path = "new_ground_truth.xlsx"
    
    try:
        qs, ans = load_ground_truth_from_excel(file_path)
        print(f"Berhasil memuat {len(qs)} pasang QA dari {file_path}")
        
        # JALANKAN EVALUASI
        evaluate_retrieval_methods(qs, ans)
        evaluate_generation_models(qs, ans)
        
    except Exception as e:
        print(f"Terjadi kesalahan fatal: {e}")