from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import threading
import re

from config import GEMINI_API_KEY
from rag_utils_baru import (
    retrieve_documents, 
    build_context_from_results, 
    get_documents_and_embeddings,
    retrieve_with_hyde
)

app = Flask(__name__)

is_kb_ready = False

def background_loader():
    """Background task to load models and embeddings."""
    global is_kb_ready
    print("[Loader] Starting background loading of knowledge base...")
    get_documents_and_embeddings() # This triggers the heavy load in rag_utils
    print("[Loader] Knowledge base loaded successfully.")
    is_kb_ready = True

threading.Thread(target=background_loader, daemon=True).start()

model_gemini = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model_gemini = genai.GenerativeModel('gemma-3-27b-it')
        print("Gemini AI Connected (Model: gemma-3-27b-it).")
    except:
        try:
            model_gemini = genai.GenerativeModel('gemini-flash-latest')
            print("Gemini AI Connected (Model: gemini-flash-latest).")
        except:
            model_gemini = genai.GenerativeModel('gemini-pro') 
            print("Gemini AI Connected (Model: gemini-pro).")
else:
    print("WARNING: No GEMINI_API_KEY found. Chatbot will run in limited mode.")

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

    sem_match = re.search(r'(?:sem[a-z]*|smt)\s*(\d+)', user_message, re.IGNORECASE)
    target_semester = sem_match.group(1) if sem_match else None
    
    k_retrieve = 60 if target_semester else 20
    
    retrieval_results = []
    use_hyde = False
    
    if model_gemini:
        try:
            retrieval_results = retrieve_with_hyde(user_message, k=k_retrieve, model_gen=model_gemini)
            if retrieval_results and len(retrieval_results) > 0:
                use_hyde = True
                print("[RETRIEVAL] Menggunakan HyDE (Hypothetical Document Embeddings)")
        except Exception as e:
            print(f"[WARNING] HyDE gagal: {e}. Fallback ke Hybrid Custom...")
            retrieval_results = []
    
    if not use_hyde or not retrieval_results:
        retrieval_results = retrieve_documents(user_message, k=k_retrieve, lexical_boost=True)
        if not use_hyde:
            print("[RETRIEVAL] Menggunakan Hybrid Custom (Dense + Lexical Boost)")
    
    if target_semester:
        filtered_results = []
        other_results = []
        for doc, score in retrieval_results:
            if f"Semester: {target_semester}" in doc or f" | {target_semester} | " in doc:
                 filtered_results.append((doc, score))
            else:
                 other_results.append((doc, score))
        
        if filtered_results:
             final_results = filtered_results + other_results[:(30 - len(filtered_results))]
             retrieval_results = final_results
        else:
             pass

    retrieved_context = build_context_from_results(retrieval_results, min_score=0.1) 
    high_relevance = any(score > 0.5 for _, score in retrieval_results)

    if model_gemini:
        try:
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

            response = model_gemini.generate_content(prompt)
            return jsonify({'response': response.text})

        except Exception as e:
            print(f"Gemini Error Detail: {e}")
            import traceback
            traceback.print_exc() # Print full stack trace
            return jsonify({'response': f"Maaf, ada kesalahan teknis: {str(e)}"})

    else:
        if high_relevance:
            best_doc, _ = retrieval_results[0]
            return jsonify({'response': f"**Info Kampus:**\n{best_doc}\n\n*(Mode terbatas: Pasang API Key untuk fitur tanya jawab bebas)*"})
        else:
            return jsonify({'response': "Maaf, saya tidak menemukan info spesifik tentang itu di database kampus. (Untuk pertanyaan umum, mohon konfigurasi API Key Gemini)."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)