import pandas as pd
import time
import os
import requests
import json
import numpy as np

# --- LIBRARY MODEL ---
import google.generativeai as genai
from groq import Groq
from openai import OpenAI  # Untuk akses OpenRouter
from huggingface_hub import InferenceClient # Untuk akses Hugging Face

# --- METRICS ---
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- LOCAL MODULES ---
try:
    from config import (
        GEMINI_API_KEY, 
        MISTRAL_API_KEY, 
        GROQ_API_KEY, 
        OPENROUTER_API_KEY, 
        HUGGINGFACE_API_KEY
    )
except ImportError:
    print("[ERROR] File config.py tidak ditemukan/tidak lengkap.")
    exit()

from rag_utils_baru import RetrievalSystem

# ==========================================
# 1. SETUP CLIENTS
# ==========================================

# A. GEMINI
model_gemini = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel('gemma-3-27b-it')
        print("[INIT] Gemini Connected.")
    except: print("[WARNING] Gemini Error.")

# B. GROQ
client_groq = None
if GROQ_API_KEY:
    try:
        client_groq = Groq(api_key=GROQ_API_KEY)
        print("[INIT] Groq Connected.")
    except: print("[WARNING] Groq Error.")

# C. OPENROUTER (Llama via OpenAI Client)
client_openrouter = None
if OPENROUTER_API_KEY:
    try:
        # OpenRouter menggunakan protokol yang sama dengan OpenAI
        client_openrouter = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        print("[INIT] OpenRouter Connected.")
    except: print("[WARNING] OpenRouter Error.")

# D. HUGGING FACE HUB
client_hf = None
if HUGGINGFACE_API_KEY:
    try:
        client_hf = InferenceClient(token=HUGGINGFACE_API_KEY)
        print("[INIT] HuggingFace Connected.")
    except: print("[WARNING] HuggingFace Error.")


# ==========================================
# 2. DEFINISI MODEL
# ==========================================
MISTRAL_MODEL_ID = "mistral-tiny"

GROQ_MODELS = {
    "Llama3": "llama3-8b-8192",
    "Qwen": "qwen-2.5-72b-instruct",
}

# Model Llama via OpenRouter (Bisa ganti 'meta-llama/llama-3-8b-instruct:free' jika ada)
OPENROUTER_MODEL = "meta-llama/llama-3-8b-instruct"

# Model via Hugging Face (Phi-3 Mini sangat cepat & ringan untuk API gratis)
HF_MODEL = "microsoft/Phi-3-mini-4k-instruct"


# ==========================================
# 3. FUNGSI PEMANGGIL MODEL
# ==========================================

def call_gemini(prompt):
    if not model_gemini: return "[ERROR] Client Missing"
    try:
        return model_gemini.generate_content(prompt).text.strip()
    except Exception as e: return f"[ERROR] Gemini: {e}"

def call_groq(prompt, model_id):
    if not client_groq: return "[ERROR] Client Missing"
    try:
        resp = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e: return f"[ERROR] Groq: {e}"

def call_mistral(prompt):
    if not MISTRAL_API_KEY: return "[ERROR] Key Missing"
    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": MISTRAL_MODEL_ID, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e: return f"[ERROR] Mistral: {e}"

def call_openrouter(prompt):
    """Akses Llama via OpenRouter menggunakan OpenAI Client"""
    if not client_openrouter: return "[ERROR] Client Missing"
    try:
        resp = client_openrouter.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # Menambahkan header referer (disarankan OpenRouter)
            extra_headers={"HTTP-Referer": "http://localhost:8000", "X-Title": "NLPEval"},
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e: return f"[ERROR] OpenRouter: {e}"

def call_huggingface(prompt):
    """Akses Model via Hugging Face Inference API"""
    if not client_hf: return "[ERROR] Client Missing"
    try:
        # Menggunakan chat_completion untuk model instruct
        resp = client_hf.chat_completion(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e: 
        return f"[ERROR] HuggingFace: {e}"

# ==========================================
# 4. UTILITIES EVALUASI & MAIN
# ==========================================

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1

def calc_metrics(ref, cand):
    if not ref or not cand or "[ERROR]" in cand: return 0.0, 0.0
    try:
        r_l = scorer.score(ref, cand)['rougeL'].fmeasure
        bleu = sentence_bleu([ref.lower().split()], cand.lower().split(), smoothing_function=smooth_fn)
        return r_l, bleu
    except: return 0.0, 0.0

def parse_ids(raw):
    s = str(raw).replace('.', ',').replace('.0', '')
    return [int(p) for p in s.split(',') if p.strip().isdigit()]

def get_ref_text(engine, t_ids, t_type):
    texts = []
    for d in engine.documents:
        try:
            if str(d['source_type']).strip() == str(t_type).strip() and \
               int(float(str(d['source_id']))) in t_ids:
                texts.append(d['text'])
        except: continue
    return " ".join(texts)

def main():
    DATA_FILE = "data_baru.xlsx"
    GT_FILE = "new_ground_truth_baru.xlsx"
    OUTPUT_FILE = "hasil_evaluasi_chatbot_final.xlsx"

    print("\n=== EVALUASI CHATBOT: 5 PLATFORM ===")
    
    try:
        engine = RetrievalSystem(DATA_FILE)
        df_gt = pd.read_excel(GT_FILE, sheet_name="GT", dtype=str)
    except Exception as e:
        print(f"[STOP] Init Fail: {e}")
        return

    results = []
    
    # DAFTAR MODEL YANG DIUJI
    # (Label Laporan, Fungsi, Argumen Tambahan)
    models = [
        ("Gemini-Gemma3", call_gemini, {}),
        ("Mistral-Tiny", call_mistral, {}),
        ("Groq-Llama3", call_groq, {"model_id": GROQ_MODELS["Llama3"]}),
        ("Groq-Qwen", call_groq, {"model_id": GROQ_MODELS["Qwen"]}),
        ("OpenRouter-Llama3", call_openrouter, {}), # Model via OpenRouter
        ("HF-Phi3", call_huggingface, {})           # Model via Hugging Face
    ]

    total = len(df_gt)
    print(f"Total Query: {total}\n")

    for i, row in df_gt.iterrows():
        q = str(row['question'])
        t_ids = parse_ids(row['source_id'])
        t_type = str(row['source_type']).strip()
        
        ref = get_ref_text(engine, t_ids, t_type)
        if not ref: continue

        print(f"[{i+1}/{total}] Q: {q[:30]}...")

        # RETRIEVAL (VSM Cosine)
        docs = engine.search(q, method='vsm_cosine', top_k=3)
        ctx = "\n".join([f"- {d['text']}" for d in docs])
        
        prompt = f"Konteks:\n{ctx}\n\nPertanyaan: {q}\nJawab singkat berdasarkan konteks."

        for label, func, kwargs in models:
            start = time.time()
            try:
                ans = func(prompt, **kwargs)
            except: ans = "[ERROR] Crash"
            dur = time.time() - start
            
            rl, bleu = calc_metrics(ref, ans)
            print(f"   -> {label}: ROUGE={rl:.2f} BLEU={bleu:.2f} ({dur:.1f}s)")
            
            results.append({
                'No': i+1, 'Query': q, 'Model': label, 
                'Answer': ans, 'Ref': ref[:100], 
                'ROUGE-L': rl, 'BLEU': bleu, 'Time': dur
            })
            time.sleep(1) # Rate limit safety

    if results:
        df = pd.DataFrame(results)
        summ = df.groupby('Model')[['ROUGE-L', 'BLEU', 'Time']].mean().sort_values('ROUGE-L', ascending=False)
        print("\n=== RANKING MODEL ===")
        print(summ)
        with pd.ExcelWriter(OUTPUT_FILE) as w:
            df.to_excel(w, sheet_name='Detail', index=False)
            summ.to_excel(w, sheet_name='Summary')
        print(f"\nSaved to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()