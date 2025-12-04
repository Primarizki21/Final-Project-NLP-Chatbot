from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from openai import OpenAI
import os

# Import konfigurasi dan retrieval logic
from config import (
    GEMINI_API_KEY, 
    OPENAI_API_KEY, 
    DEEPSEEK_API_KEY, 
    OPENROUTER_API_KEY,
    CURRENT_RETRIEVAL_METHOD, 
    CURRENT_GENERATION_MODEL
)
from rag_utils import retrieve_documents, build_context_from_results

app = Flask(__name__)

# --- SETUP CLIENTS ---

# 1. Google Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# 2. OpenAI (untuk GPT-4o Mini)
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
        base_url="https://openrouter.ai/api/v1",
    )

def generate_answer(query, context, model_name):
    """
    Generator jawaban dengan switch logic model (Gemini, GPT, DeepSeek, OpenRouter)
    """
    system_instruction = f"""
    Anda adalah Asisten Akademik. Jawab pertanyaan berdasarkan KONTEKS berikut.
    Jika tidak ada info di konteks, katakan tidak tahu.
    
    KONTEKS:
    {context}
    """

    try:
        # A. GEMINI
        if model_name == 'gemini':
            if not GEMINI_API_KEY: raise Exception("GEMINI_API_KEY tidak ditemukan.")
            full_prompt = f"{system_instruction}\n\nPertanyaan: {query}"
            response = gemini_model.generate_content(full_prompt)
            return response.text

        # B. GPT-4o MINI
        elif model_name == 'gpt4o':
            if not client_openai: raise Exception("OPENAI_API_KEY tidak ditemukan atau salah.")
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content

        # C. DEEPSEEK
        elif model_name == 'deepseek':
            if not client_deepseek: raise Exception("DEEPSEEK_API_KEY tidak ditemukan.")
            response = client_deepseek.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query}
                ],
                stream=False
            )
            return response.choices[0].message.content

        # D. OPENROUTER
        elif model_name == 'openrouter':
            if not client_openrouter: raise Exception("OPENROUTER_API_KEY tidak ditemukan.")
            # Menggunakan model gratis Llama 3.1 8B via OpenRouter
            response = client_openrouter.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free", 
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content

    except Exception as e:
        error_msg = f"Error pada Model {model_name}: {str(e)}"
        # Print error ke terminal server agar bisa didebug
        print(f"\n[SERVER LOG] Gagal Generate Answer: {error_msg}\n")
        return error_msg

    return "Model tidak dikenali atau Client belum siap."

@app.route('/')
def home():
    # Pastikan ada file templates/index.html
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message: 
        return jsonify({'response': "Pesan kosong."})

    # 1. Retrieval
    # Menggunakan metode dari config (dense, hybrid, rerank, jaccard, tfidf, doc2vec)
    docs = retrieve_documents(user_message, k=5, method=CURRENT_RETRIEVAL_METHOD)
    context = build_context_from_results(docs)

    # 2. Generation
    # Menggunakan model dari config (gemini, gpt4o, deepseek, openrouter)
    answer = generate_answer(user_message, context, CURRENT_GENERATION_MODEL)
    
    # Info debug untuk UI (opsional)
    debug_info = f"\n<br><small style='color:gray'>Retrieval: {CURRENT_RETRIEVAL_METHOD} | Model: {CURRENT_GENERATION_MODEL}</small>"

    return jsonify({'response': answer + debug_info})

if __name__ == '__main__':
    app.run(debug=True, port=5000)