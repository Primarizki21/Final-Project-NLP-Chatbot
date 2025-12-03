# app.py
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import cohere
from groq import Groq
import os
from config import GEMINI_API_KEY, COHERE_API_KEY, GROQ_API_KEY

# Import fungsi retrieval baru kita
from rag_utils import retrieve_documents, build_context_from_results

app = Flask(__name__)

# --- SETUP CLIENTS ---
# 1. Gemini Client
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# 2. Cohere Client
co = cohere.Client(COHERE_API_KEY)

# 3. Groq Client (untuk Llama 3)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- PENGATURAN EKSPERIMEN ---
CURRENT_RETRIEVAL_METHOD = 'rerank'
CURRENT_GENERATION_MODEL = 'llama'

def generate_answer(query, context, model_name):
    """Fungsi sentral untuk switch model generation"""
    
    system_prompt = f"""
    Anda adalah asisten akademik Prodi Teknologi Sains Data.
    Jawab pertanyaan berdasarkan Konteks berikut. Jika tidak ada di konteks, jawab tidak tahu.
    
    Konteks:
    {context}
    
    Pertanyaan: {query}
    """
    
    # A. MODEL GEMINI 2.0 FLASH
    if model_name == 'gemini':
        response = gemini_model.generate_content(system_prompt)
        return response.text

    # B. MODEL COHERE COMMAND R
    elif model_name == 'cohere':
        response = co.chat(
            message=query,
            documents=[{"text": context}], # Cohere punya fitur RAG native
            model='command-r', 
            temperature=0.3
        )
        return response.text

    # C. MODEL LLAMA 3.3 70B (via Groq)
    elif model_name == 'llama':
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Anda adalah asisten dosen wali yang membantu."},
                {"role": "user", "content": system_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    
    return "Model not configured."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message: return jsonify({'response': "Error."})

    try:
        # 1. Retrieval (Menggunakan metode yang dipilih di config atas)
        # Note: 'rerank' mungkin lebih lambat sedikit dibanding 'dense'
        docs = retrieve_documents(user_message, k=5, method=CURRENT_RETRIEVAL_METHOD)
        
        # Ambang batas score (threshold) mungkin perlu disesuaikan per metode
        # Untuk Hybrid/Rerank, scorenya mungkin berbeda range-nya.
        context = build_context_from_results(docs, min_score=-100) # Ambil semua top-k

        # 2. Generation (Menggunakan model yang dipilih di config atas)
        answer = generate_answer(user_message, context, CURRENT_GENERATION_MODEL)
        
        # Tambahkan info debug (opsional, biar kelihatan di UI metode apa yang dipakai)
        debug_info = f"\n\n*(Metode: {CURRENT_RETRIEVAL_METHOD} | Model: {CURRENT_GENERATION_MODEL})*"
        
        return jsonify({'response': answer + debug_info})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': f"Terjadi kesalahan: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)