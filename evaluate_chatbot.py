"""
Skrip Evaluasi Chatbot (Revised dengan Debugging)
Fitur:
1. Membandingkan 6 Metode Retrieval: Dense, Hybrid, Rerank, Jaccard, TF-IDF, Doc2Vec.
2. Membandingkan 4 Model Generation: Gemini, GPT-4o Mini, DeepSeek, OpenRouter.
3. Fitur Debug Print untuk melihat error API secara langsung.
"""

import os
import pandas as pd
import numpy as np
from typing import List
from rouge_score import rouge_scorer
from sentence_transformers import util

# API Clients
import google.generativeai as genai
from openai import OpenAI

# Import Local - Pastikan file rag_utils.py dan config.py sudah benar
from rag_utils import (
    retrieve_documents,
    get_retrieval_model_objects,
    build_context_from_results,
    initialize_retrieval_system
)
from config import (
    GEMINI_API_KEY, 
    OPENAI_API_KEY, 
    DEEPSEEK_API_KEY, 
    OPENROUTER_API_KEY
)

# --- SETUP CLIENTS ---
# 1. Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 2. OpenAI (GPT-4o Mini)
client_openai = None
if OPENAI_API_KEY:
    client_openai = OpenAI(api_key=OPENAI_API_KEY)

# 3. DeepSeek (via OpenAI Client)
client_deepseek = None
if DEEPSEEK_API_KEY:
    client_deepseek = OpenAI(
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com"
    )

# 4. OpenRouter (via OpenAI Client)
client_openrouter = None
if OPENROUTER_API_KEY:
    client_openrouter = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

def load_ground_truth(path: str = "new_ground_truth.xlsx") -> tuple[list, list]:
    """Load QA pairs dari Excel."""
    if not os.path.exists(path):
        print(f"[WARNING] File {path} tidak ditemukan. Membuat dummy data untuk tes.")
        return ["Apa syarat skripsi?", "Siapa kaprodi TSD?"], ["Minimal 144 SKS.", "Pak Ardi."]
        
    df = pd.read_excel(path)
    # Cari kolom yang mirip 'question' dan 'answer' secara fleksibel
    q_col = next((c for c in df.columns if 'tanya' in c.lower() or 'quest' in c.lower()), None)
    a_col = next((c for c in df.columns if 'jawab' in c.lower() or 'ans' in c.lower()), None)
    
    if not q_col or not a_col:
        print("[WARNING] Kolom pertanyaan/jawaban tidak ditemukan. Menggunakan kolom index 0 dan 1.")
        q_col = df.columns[0]
        a_col = df.columns[1]
    
    questions = df[q_col].dropna().astype(str).tolist()
    answers = df[a_col].dropna().astype(str).tolist()
    
    limit = min(len(questions), len(answers))
    return questions[:limit], answers[:limit]

def evaluate_retrieval_methods(questions, answers):
    print("\n" + "="*50)
    print("EVALUASI RETRIEVAL (6 METODE)")
    print("="*50)

    # Inisialisasi sistem sekali saja
    embedder, documents, doc_embeddings = get_retrieval_model_objects()
    if not documents: 
        print("Dokumen kosong, skip retrieval evaluation.")
        return

    # Hitung embedding jawaban asli (Ground Truth) sebagai acuan relevansi
    print("Menghitung embedding Ground Truth...")
    gt_embeddings = embedder.encode(answers, convert_to_tensor=True)

    methods = ['dense', 'hybrid', 'rerank', 'jaccard', 'tfidf', 'doc2vec']
    results_summary = []

    for method in methods:
        print(f"\nTesting Metode: {method.upper()} ...")
        precisions, recalls = [], []
        k = 5
        threshold = 0.5 # Ambang batas kemiripan jawaban asli dengan dokumen DB (Cosine Sim)

        for i, (q, gt_emb) in enumerate(zip(questions, gt_embeddings)):
            # 1. Oracle: Tentukan dokumen mana di DB yang SEHARUSNYA muncul
            sim_scores = util.cos_sim(gt_emb, doc_embeddings)[0]
            relevant_indices = [idx for idx, s in enumerate(sim_scores) if s >= threshold]

            # Jika pertanyaan ini tidak punya dokumen relevan di DB (fakta baru), skip
            if not relevant_indices: continue

            # 2. Lakukan Retrieval Aktual
            retrieved = retrieve_documents(q, k=k, method=method)
            
            # 3. Hitung Intersection
            hits = 0
            for doc_text, _ in retrieved:
                if doc_text in documents:
                    idx = documents.index(doc_text)
                    if idx in relevant_indices:
                        hits += 1
            
            p = hits / k
            r = hits / len(relevant_indices)
            precisions.append(p)
            recalls.append(r)

        avg_p = np.mean(precisions) if precisions else 0
        avg_r = np.mean(recalls) if recalls else 0
        print(f"  -> Precision: {avg_p:.4f}, Recall: {avg_r:.4f}")
        
        results_summary.append({
            "Method": method, 
            "Avg Precision": avg_p, 
            "Avg Recall": avg_r
        })

    # Simpan hasil
    try:
        pd.DataFrame(results_summary).to_csv("hasil_evaluasi_retrieval.csv", index=False)
        print("\nHasil Retrieval disimpan ke 'hasil_evaluasi_retrieval.csv'")
    except Exception as e:
        print(f"Gagal menyimpan CSV: {e}")

def generate_answer_eval(model_name, prompt):
    """Fungsi helper generation dengan error handling ketat."""
    try:
        if model_name == 'gemini':
            if not GEMINI_API_KEY: return "Error: GEMINI_API_KEY missing"
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content(prompt).text.strip()
        
        elif model_name == 'gpt4o':
            if not client_openai: return "Error: OPENAI_API_KEY missing or invalid"
            res = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content.strip()
            
        elif model_name == 'deepseek':
            if not client_deepseek: return "Error: DEEPSEEK_API_KEY missing"
            res = client_deepseek.chat.completions.create(
                model="deepseek-chat", # Cek apakah nama modelnya benar (kadang 'deepseek-coder')
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content.strip()
            
        elif model_name == 'openrouter':
            if not client_openrouter: return "Error: OPENROUTER_API_KEY missing"
            # Gunakan model gratisan yang reliable di OpenRouter
            res = client_openrouter.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content.strip()
            
    except Exception as e:
        # Return pesan error asli agar bisa di-debug
        return f"API Error: {str(e)}"
    
    return "Skipped (Unknown Model)"

def evaluate_generation_models(questions, answers):
    print("\n" + "="*50)
    print("EVALUASI GENERATION (ROUGE SCORE)")
    print("="*50)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    
    # List model yang akan dites
    models_to_test = []
    if GEMINI_API_KEY: models_to_test.append('gemini')
    if OPENAI_API_KEY: models_to_test.append('gpt4o')
    if DEEPSEEK_API_KEY: models_to_test.append('deepseek')
    if OPENROUTER_API_KEY: models_to_test.append('openrouter')

    if not models_to_test:
        print("Tidak ada API Key yang terdeteksi di config.py.")
        return

    # Gunakan 1 metode retrieval terbaik (misal 'rerank') untuk semua model agar adil
    best_retrieval = 'rerank' 
    
    # Batasi sampel agar tidak boros kuota API saat testing
    MAX_SAMPLES = 5 
    print(f"Evaluasi dilakukan pada {MAX_SAMPLES} sampel pertanyaan per model.")

    qs_subset = questions[:MAX_SAMPLES]
    ans_subset = answers[:MAX_SAMPLES]

    summary_rouge = []

    for model_name in models_to_test:
        print(f"\nTesting Model: {model_name.upper()} ...")
        r1_scores, rl_scores = [], []

        for i, (q, gt) in enumerate(zip(qs_subset, ans_subset)):
            # 1. Retrieve context
            # (Pastikan rag_utils sudah meload dokumen dengan benar)
            docs = retrieve_documents(q, k=5, method=best_retrieval)
            context = build_context_from_results(docs)
            
            prompt = f"Context: {context}\n\nQuestion: {q}\n\nAnswer in Indonesian:"
            
            # 2. Generate
            pred = generate_answer_eval(model_name, prompt)
            
            # --- DEBUGGING OUTPUT ---
            if "Error" in pred or "Skipped" in pred:
                print(f"  [DEBUG] Gagal Sample {i+1}: {pred}")
            else:
                # Hanya hitung skor jika berhasil generate
                s = scorer.score(gt, pred)
                r1_scores.append(s["rouge1"].fmeasure)
                rl_scores.append(s["rougeL"].fmeasure)
                # print(f"  [OK] Sample {i+1} Success") # Uncomment jika ingin lebih verbose

        # Hitung rata-rata
        if r1_scores:
            avg_r1 = np.mean(r1_scores)
            avg_rl = np.mean(rl_scores)
        else:
            avg_r1 = 0.0
            avg_rl = 0.0
            print(f"  [WARNING] Semua request gagal untuk model {model_name}. Cek API Key/Quota.")
        
        print(f"  -> ROUGE-1: {avg_r1:.4f}")
        
        summary_rouge.append({
            "Model": model_name,
            "Avg ROUGE-1": avg_r1,
            "Avg ROUGE-L": avg_rl
        })

    # Simpan hasil
    try:
        pd.DataFrame(summary_rouge).to_csv("hasil_evaluasi_generation.csv", index=False)
        print("\nHasil Generation disimpan ke 'hasil_evaluasi_generation.csv'")
    except Exception as e:
        print(f"Gagal menyimpan CSV: {e}")

if __name__ == "__main__":
    qs, ans = load_ground_truth()
    if qs:
        # evaluate_retrieval_methods(qs, ans) # Uncomment untuk jalankan retrieval eval
        evaluate_generation_models(qs, ans)