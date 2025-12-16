import os
import threading
import time
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

from rag_utils_baru import RetrievalSystem

app = Flask(__name__)

is_kb_ready = False
retriever = None
model_gemini = None

try:
    from config import GEMINI_API_KEY
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel('gemma-3-27b-it')
        print("[INIT] Gemini AI terhubung.")
except ImportError:
    print("[WARNING] config.py tidak ditemukan.")
    GEMINI_API_KEY = None

def background_loader():
    global is_kb_ready, retriever
    print("[LOADER] Memuat knowledge base...")
    try:
        retriever = RetrievalSystem("data_baru.xlsx")
        print("[LOADER] Knowledge base siap.")
        is_kb_ready = True
    except Exception as e:
        print(f"[ERROR] Gagal memuat knowledge base: {e}")

threading.Thread(target=background_loader, daemon=True).start()

def retrieve_documents(query, k=20):
    if not retriever:
        return []
    return retriever.search(query, method='bm25', top_k=k)

def build_context(results, min_score=0.5):
    if not results:
        return "Tidak ada informasi relevan ditemukan."
    
    context_parts = []
    for i, doc in enumerate(results[:15]):
        if doc['score'] >= min_score:
            context_parts.append(f"[{i+1}] {doc['text']}")
    
    return "\n\n".join(context_parts) if context_parts else "Tidak ada informasi relevan."

@app.route('/')
def home():
    if not is_kb_ready:
        return render_template('loading.html')
    return render_template('index.html')

@app.route('/check_status')
def check_status():
    return jsonify({'ready': is_kb_ready})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', [])

    if not user_message:
        return jsonify({'response': "Silakan ketik sesuatu."})

    retrieval_results = retrieve_documents(user_message, k=20)
    retrieved_context = build_context(retrieval_results)

    history_text = ""
    for msg in history[-40:]:
        role_label = "User" if msg.get('role') == 'user' else "Assistant"
        history_text += f"{role_label}: {msg.get('text', '')}\n"

    prompt = f"""
    You are a smart and helpful academic assistant for Data Science students at Universitas Airlangga.

    Conversation History:
    {history_text}

    Current User Question: "{user_message}"

    Context from Curriculum/Handbook (might be relevant):
    {retrieved_context}

    INSTRUCTIONS:
    1. **PRIORITY 1 - CHECK INTENT:** Determine if the user's input is "General Conversation/Greeting" OR "Academic Question".
    - **General Conversation (Greeting, Small Talk, Games, Jokes):** IGNORE the context. Answer naturally and friendly in Indonesian. DO NOT mention the syllabus or documents. Example: "Halo! Ada yang bisa saya bantu?" or playing along with games.
    - **Academic Question (Course, Curriculum, Rules, Campus):** Use the 'Context from Curriculum/Handbook'. Answer faithfully based on the context. If the info is NOT in the context, say "Informasi tersebut tidak tercantum dalam silabus/dokumen".

    2. **PRIORITY 2 - CONTEXTUAL CONTINUITY:** If the user follows up on a previous topic (based on History), maintain that context.

    3. If the question asks for a LIST (e.g. "apa saja mata kuliah di semester 1"), combine information from ALL relevant context snippets.
    4. Answer DIRECTLY.
    5. Use Indonesian language (Bahasa Indonesia).
    6. **FORMATTING:** Use markdown formatting consistently:
    - Use **bold** (double asterisk) for course names and important terms
    - Use *italic* (single asterisk) ONLY if needed for emphasis, but be consistent
    - For lists, use bullet points with "- " or "* " (single asterisk with space) - do NOT use *italic* format for list items
    - Example: "**Manajemen Jaringan** (SIJ303) - 3 sks" (bold for course name, not italic)
    7. **CRITICAL - NO PLACEHOLDERS:** 
    - ALWAYS use the actual course names from the context. DO NOT use placeholders like __MD0___, __MD1___, etc.
    - If you see course information in the context, extract and use the REAL course name (e.g., "Manajemen Jaringan", "Multivariat", "Data Mining II")
    - Format: "* **Course Name** (CODE) - X sks" for list items
    - Example: "* **Manajemen Jaringan** (SIJ303) - 3 sks" NOT "* __MD0___ (SIJ303) - 3 sks"
    """

    try:
        if model_gemini:
            response = model_gemini.generate_content(prompt)
            final_answer = response.text
        else:
            final_answer = "Fitur AI belum tersedia. Mode terbatas."

        return jsonify({'response': final_answer})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'response': "Terjadi kesalahan pada server."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)