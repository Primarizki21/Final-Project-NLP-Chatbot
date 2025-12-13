import os
import time
import pandas as pd
import google.generativeai as genai
from rouge_score import rouge_scorer
from rag_utils_baru import retrieve_documents, get_documents_and_embeddings
from config import GEMINI_API_KEY

# --- CONFIGURATION ---
# --- CONFIGURATION ---
DATASET_PATH = "new_ground_truth.xlsx"
SAMPLE_SIZE = 100  # User specified 10 samples in the context
K_RETRIEVAL = 3
CONTEXT_MIN_SCORE = 0.2
OUTPUT_FILE = "evaluation_generation_3_methods_v2.xlsx"

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("ERROR: API Key is missing.")
    exit()

def get_model(model_name="gemma-3-27b-it"):
    """Helper to initialize model."""
    print(f"Initializing model: {model_name}...")
    return genai.GenerativeModel(model_name)

# Initial Model (Use Gemma to bypass Gemini quotas)
model = get_model("gemma-3-27b-it") 

# --- HELPER FUNCTIONS ---

def load_dataset(path: str, limit: int):
    """Memuat dataset pertanyaan dan jawaban (Ground Truth)."""
    try:
        df = pd.read_excel(path)
        cols = df.columns
        q_col = next((c for c in ["question", "pertanyaan", "Question"] if c in cols), None)
        a_col = next((c for c in ["answer", "jawaban", "Answer"] if c in cols), None)
        
        if not q_col or not a_col:
            print("Kolom Question/Answer tidak ditemukan di dataset.")
            return [], []
            
        questions = [str(x).strip() for x in df[q_col].dropna().tolist()]
        answers = [str(x).strip() for x in df[a_col].dropna().tolist()]
        
        n = min(len(questions), len(answers), limit)
        return questions[:n], answers[:n]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], []

def retrieve_lexical_only(query: str, documents: list, k: int = 5):
    """
    Metode 1: Lexical Retrieval (Keyword Matching).
    Diambil dari test_retrieval_50.py
    """
    query_tokens = set(query.lower().split())
    scores = []
    
    for doc in documents:
        doc_tokens = set(doc.lower().split())
        if not doc_tokens:
            scores.append(0.0)
            continue
        overlap = len(query_tokens.intersection(doc_tokens))
        score = overlap / len(query_tokens) if query_tokens else 0
        scores.append(score)
    
    # Sort descending
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs[:k]

def build_context(results, min_score=0.0):
    """Helper untuk menyusun string konteks dari hasil retrieval."""
    context_parts = []
    for doc, score in results:
        if score >= min_score:
            context_parts.append(f"- {doc}")
    return "\n".join(context_parts)

def generate_with_retry(prompt):
    """Generate content dengan mekanisme retry sederhana."""
    if not prompt:
        print("[DEBUG] Prompt kosong!")
        return ""
        
    try:
        response = model.generate_content(prompt)
        # Check if response has parts or text
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'parts'):
            return response.parts[0].text.strip()
        else:
            print(f"[DEBUG] Response object has no text attribute: {response}")
            return ""
            
    except Exception as e:
        if "429" in str(e): # Rate limit
            print("[DEBUG] Rate limit hit. Waiting 2s...")
            time.sleep(2)
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e2:
                print(f"[DEBUG] Retry failed: {e2}")
                return ""
        
        # Print FULL error details for the user
        print(f"\n[DEBUG] Gemini Error: {e}")
        if hasattr(e, 'args'):
             print(f"[DEBUG] Args: {e.args}")
        return ""

# --- MAIN EXECUTION ---

def main():
    global model
    print("=== EVALUASI ANSWER GENERATION (GEMINI) DENGAN 3 METODE RETRIEVAL ===\n")

    # 0. Test Connection First (With Auto-Fallback)
    print("Mencoba koneksi ke Gemini API...")
    try:
        test_resp = model.generate_content("Hello")
        safe_text = test_resp.text.strip().encode('ascii', 'ignore').decode('ascii')
        print(f"Test Connection Result: Berhasil! ({safe_text})")
    except Exception as e:
        print(f"Warn: Model utama gagal ({e}). Mencoba fallback ke 'gemini-1.5-flash-latest'...")
        try:
            model = get_model("gemini-1.5-flash-latest")
            test_resp = model.generate_content("Hello")
            print(f"Fallback Connection Result: Berhasil! ({test_resp.text.strip()})")
        except Exception as e2:
            print(f"FATAL: Semua percobaan koneksi GAGAL. Cek API Key/Quota.\nError Terakhir: {e2}")
            return
    
    # 1. Load Data
    print(f"Loading dataset: {DATASET_PATH}...")
    questions, answers = load_dataset(DATASET_PATH, limit=SAMPLE_SIZE)
    if not questions:
        return
    print(f"Loaded {len(questions)} items.")

    # 2. Prepare Retrieval System
    print("Initializing Knowledge Base (this may take a moment)...")
    # Menggunakan fungsi dari rag_utils agar konsisten dengan files lain
    documents, _ = get_documents_and_embeddings()
    if not documents:
        print("Knowledge base kosong!")
        return
    print(f"Knowledge Base ready. Total Documents: {len(documents)}")

    # 3. Setup Metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    results = []
    
    print("\nStarting Evaluation Loop...")
    for i, (q, gt) in enumerate(zip(questions, answers)):
        print(f"[{i+1}/{len(questions)}] Processing: {q[:40]}...")
        
        # --- METODE 1: LEXICAL RETRIEVAL ---
        lex_results = retrieve_lexical_only(q, documents, k=K_RETRIEVAL)
        lex_context = build_context(lex_results, min_score=0.0) # Lexical scores are usually low (0.0-1.0)
        
        prompt_lex = f"Context:\n{lex_context}\n\nQuestion: {q}\nAnswer in Indonesian based on context."
        ans_lex = generate_with_retry(prompt_lex)
        score_lex = scorer.score(gt, ans_lex)['rougeL'].fmeasure if ans_lex else 0.0
        
        results.append({
            "Question": q,
            "Method": "Lexical",
            "Context": lex_context,
            "Generated Answer": ans_lex,
            "ROUGE-L": score_lex
        })
        time.sleep(10) # 10s delay as requested
        
        # --- METODE 2: DENSE RETRIEVAL (Semantic Only) ---
        # lexical_boost=False makes it pure dense retrieval
        dense_results = retrieve_documents(q, k=K_RETRIEVAL, lexical_boost=False)
        dense_context = build_context(dense_results, min_score=CONTEXT_MIN_SCORE)
        
        prompt_dense = f"Context:\n{dense_context}\n\nQuestion: {q}\nAnswer in Indonesian based on context."
        ans_dense = generate_with_retry(prompt_dense)
        score_dense = scorer.score(gt, ans_dense)['rougeL'].fmeasure if ans_dense else 0.0
        
        results.append({
            "Question": q,
            "Method": "Dense",
            "Context": dense_context,
            "Generated Answer": ans_dense,
            "ROUGE-L": score_dense
        })
        time.sleep(10)

        # --- METODE 3: HYBRID RETRIEVAL (Semantic + Lexical Boost) ---
        # lexical_boost=True is the default hybrid implementation in rag_utils
        hybrid_results = retrieve_documents(q, k=K_RETRIEVAL, lexical_boost=True)
        hybrid_context = build_context(hybrid_results, min_score=CONTEXT_MIN_SCORE)
        
        prompt_hybrid = f"Context:\n{hybrid_context}\n\nQuestion: {q}\nAnswer in Indonesian based on context."
        ans_hybrid = generate_with_retry(prompt_hybrid)
        score_hybrid = scorer.score(gt, ans_hybrid)['rougeL'].fmeasure if ans_hybrid else 0.0
        
        results.append({
            "Question": q,
            "Method": "Hybrid",
            "Context": hybrid_context,
            "Generated Answer": ans_hybrid,
            "ROUGE-L": score_hybrid
        })
        time.sleep(10)

    # 4. Save & Summary
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("RATA-RATA SKOR ROUGE-L PER METODE")
    print("="*50)
    summary = df_res.groupby("Method")["ROUGE-L"].mean()
    print(summary)
    
    df_res.to_excel(OUTPUT_FILE, index=False)
    print(f"\nDisimpan ke: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
