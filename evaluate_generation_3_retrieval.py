import os
import time
import pandas as pd
import google.generativeai as genai
from rouge_score import rouge_scorer
from config import GEMINI_API_KEY

# Import fungsi retrieval baru dari rag_utils_baru
from rag_utils_baru import (
    retrieve_documents, 
    retrieve_bm25, 
    retrieve_hybrid_rrf, 
    retrieve_with_rerank,
    get_documents_and_embeddings
)

# --- CONFIGURATION ---
DATASET_PATH = "new_ground_truth.xlsx"
SAMPLE_SIZE = 100 
K_RETRIEVAL = 3
CONTEXT_MIN_SCORE = 0.0
OUTPUT_FILE = "evaluation_8_methods_ir.xlsx"

# Configure Gemini
print(GEMINI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("ERROR: API Key is missing.")
    exit()

def get_model(model_name="gemma-3-27b-it"):
    return genai.GenerativeModel(model_name)

model = get_model()

def load_dataset(path: str, limit: int):
    try:
        df = pd.read_excel(path)
        cols = df.columns
        q_col = next((c for c in ["question", "pertanyaan", "Question"] if c in cols), None)
        a_col = next((c for c in ["answer", "jawaban", "Answer"] if c in cols), None)
        
        if not q_col or not a_col: return [], []
            
        questions = [str(x).strip() for x in df[q_col].dropna().tolist()]
        answers = [str(x).strip() for x in df[a_col].dropna().tolist()]
        
        n = min(len(questions), len(answers), limit)
        return questions[:n], answers[:n]
    except Exception as e:
        print(f"Error: {e}")
        return [], []

def retrieve_lexical_only(query: str, documents: list, k: int = 5):
    """Metode 1: Naive Token Overlap (Lama)"""
    query_tokens = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_tokens = set(doc.lower().split())
        overlap = len(query_tokens.intersection(doc_tokens))
        score = overlap / len(query_tokens) if query_tokens else 0
        scores.append(score)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:k]

def build_context(results):
    return "\n".join([f"- {doc}" for doc, _ in results])

def generate_with_retry(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else ""
    except Exception as e:
        if "429" in str(e):
            time.sleep(5)
            try: return model.generate_content(prompt).text.strip()
            except: return ""
        print(f"Error Gen: {e}")
        return ""

def generate_hypothetical_answer(query):
    """Untuk Metode 7: HyDE"""
    prompt = f"Tuliskan paragraf jawaban singkat yang relevan untuk pertanyaan ini (berpura-puralah tahu jawabannya):\nPertanyaan: {query}\nJawaban:"
    return generate_with_retry(prompt)

def generate_multi_queries(query):
    """Untuk Metode 8: Multi-Query"""
    prompt = f"Buatlah 3 variasi pertanyaan berbeda yang memiliki maksud sama dengan pertanyaan ini. Pisahkan dengan baris baru.\nPertanyaan: {query}"
    res = generate_with_retry(prompt)
    variations = [line.strip() for line in res.split('\n') if line.strip()]
    return variations[:3] if variations else [query]

def main():
    global model
    print("=== EVALUASI 8 METODE RETRIEVAL ===\n")

    # Load Data
    questions, answers = load_dataset(DATASET_PATH, limit=SAMPLE_SIZE)
    if not questions: return

    # Init Knowledge Base
    documents, _ = get_documents_and_embeddings()
    print(f"Knowledge Base ready. Docs: {len(documents)}")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    
    for i, (q, gt) in enumerate(zip(questions, answers)):
        print(f"\n[{i+1}/{len(questions)}] Q: {q[:40]}...")
        
        # Helper umum untuk evaluasi per metode
        def evaluate_method(method_name, context_docs):
            context_str = build_context(context_docs)
            prompt = f"Context:\n{context_str}\n\nQuestion: {q}\nAnswer in Indonesian based on context."
            ans = generate_with_retry(prompt)
            score = scorer.score(gt, ans)['rougeL'].fmeasure if ans else 0.0
            results.append({
                "Question": q,
                "Method": method_name,
                "Context": context_str[:200] + "...",
                "Generated Answer": ans,
                "ROUGE-L": score
            })
            print(f"   > {method_name}: {score:.4f}")
            time.sleep(2) # Delay kecil antar metode

        # 1. NAIVE LEXICAL (Old)
        evaluate_method("1. Naive Lexical", retrieve_lexical_only(q, documents, k=K_RETRIEVAL))

        # 2. DENSE (Old)
        evaluate_method("2. Dense (SBERT)", retrieve_documents(q, k=K_RETRIEVAL, lexical_boost=False))

        # 3. HYBRID CUSTOM (Old)
        evaluate_method("3. Hybrid Custom", retrieve_documents(q, k=K_RETRIEVAL, lexical_boost=True))

        # 4. BM25 (New - Better Lexical)
        evaluate_method("4. BM25", retrieve_bm25(q, k=K_RETRIEVAL))

        # 5. HYBRID RRF (New - BM25 + Dense fused)
        evaluate_method("5. Hybrid RRF", retrieve_hybrid_rrf(q, k=K_RETRIEVAL))

        # 6. RE-RANKING (New - Cross Encoder)
        evaluate_method("6. Re-Ranking", retrieve_with_rerank(q, k=K_RETRIEVAL))

        # 7. HyDE (New - Generative)
        hypo_answer = generate_hypothetical_answer(q)
        # Kita retrieve menggunakan jawaban palsu, bukan pertanyaan
        hyde_docs = retrieve_documents(hypo_answer, k=K_RETRIEVAL, lexical_boost=False)
        evaluate_method("7. HyDE", hyde_docs)

        # 8. MULTI-QUERY (New - Query Expansion)
        variations = generate_multi_queries(q)
        all_docs = []
        # Retrieve untuk setiap variasi
        for var_q in variations:
            all_docs.extend(retrieve_documents(var_q, k=2, lexical_boost=False)) # Ambil 2 per variasi
        # Deduplikasi berdasarkan konten teks
        seen = set()
        unique_docs = []
        for doc, score in all_docs:
            if doc not in seen:
                unique_docs.append((doc, score))
                seen.add(doc)
        evaluate_method("8. Multi-Query", unique_docs[:K_RETRIEVAL])

        time.sleep(5) # Delay antar pertanyaan

    # Summary
    df_res = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RATA-RATA SKOR ROUGE-L PER METODE")
    print("="*50)
    print(df_res.groupby("Method")["ROUGE-L"].mean())
    
    df_res.to_excel(OUTPUT_FILE, index=False)
    print(f"\nDisimpan ke: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
