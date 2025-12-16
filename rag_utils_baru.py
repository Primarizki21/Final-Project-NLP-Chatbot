import pandas as pd
import numpy as np
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

class RetrievalSystem:
    def __init__(self, data_path: str):
        self.documents = []  
        self.corpus_text = [] 
        self.tokenized_corpus = [] 
        
        # 1. Load Data
        self._load_data(data_path)
        
        if not self.documents:
            raise ValueError("Tidak ada dokumen yang berhasil dimuat. Cek file Excel.")

        # 2. Build BM25 Index
        print("[INIT] Membangun index BM25...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 3. Build VSM Cosine Index (TF-IDF dengan L2 Norm)
        print("[INIT] Membangun index VSM Cosine...")
        self.tfidf_cosine_vec = TfidfVectorizer(norm='l2', smooth_idf=True)
        self.tfidf_cosine_matrix = self.tfidf_cosine_vec.fit_transform(self.corpus_text)
        
        # 4. Build TF-IDF Sum Index (TF-IDF tanpa Norm -> Dot Product = Sum of Weights)
        print("[INIT] Membangun index TF-IDF Sum...")
        self.tfidf_sum_vec = TfidfVectorizer(norm=None, smooth_idf=True)
        self.tfidf_sum_matrix = self.tfidf_sum_vec.fit_transform(self.corpus_text)

        print("[INIT] Sistem Siap.")

    def _clean_id(self, val):
        val = str(val).strip()
        if val.endswith('.0'):
            val = val[:-2]
        return val

    def _preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_data(self, path: str):
        print(f"[LOAD] Membaca data dari {path}...")
        try:
            xls = pd.read_excel(path, sheet_name=None, dtype=str)
            
            for sheet_name, df in xls.items():
                if df.empty: continue
                
                # 1. Bersihkan Nama Kolom
                # Lowercase dan strip whitespace agar konsisten
                df.columns = [str(c).strip() for c in df.columns]
                
                for _, row in df.iterrows():
                    # --- A. Ambil ID (Metadata) ---
                    # Cari kolom yang namanya mengandung 'id' atau 'no' (case insensitive)
                    # Karena nama kolom sudah kita bersihkan di atas, kita cek manual
                    raw_id = None
                    if 'id' in df.columns:
                        raw_id = row['id']
                    elif 'no' in df.columns:
                        raw_id = row['no']
                    
                    if raw_id is None or pd.isna(raw_id): continue
                    doc_id = self._clean_id(raw_id)
                    
                    # --- B. Susun Teks Dokumen (Key: Value) ---
                    parts = []
                    
                    # row.items() akan mengembalikan (nama_kolom, nilai_sel)
                    for col_name, val in row.items():
                        # Cek validitas data
                        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                            continue
                            
                        # Format: "Nama Kolom: Isi Data"
                        # col_name kita buat Title Case biar rapi (misal: "kode mata kuliah" -> "Kode Mata Kuliah")
                        clean_col = str(col_name).replace('_', ' ').title()
                        clean_val = str(val).strip()
                        
                        # Gabungkan menjadi satu string kecil
                        parts.append(f"{clean_col}: {clean_val}")
                    
                    # Gabungkan semua bagian dengan separator " | "
                    # Separator pipa (|) bagus untuk membedakan antar field secara visual dan semantik
                    full_text = " | ".join(parts)
                    
                    # Skip jika terlalu pendek
                    if len(full_text) < 5: continue 

                    processed = self._preprocess(full_text)
                    tokens = processed.split()

                    self.documents.append({
                        'source_id': doc_id,
                        'source_type': sheet_name.strip(),
                        'text': full_text  # Hasilnya akan sangat deskriptif
                    })
                    self.corpus_text.append(processed)
                    self.tokenized_corpus.append(tokens)
                        
            print(f"[LOAD] Total dokumen: {len(self.documents)}")
            
        except Exception as e:
            print(f"[ERROR] Gagal memuat data: {e}")
            raise e
        
    def search(self, query: str, method: str, top_k: int = 10) -> List[Dict]:
        query_proc = self._preprocess(query)
        query_tokens = query_proc.split()
        
        if not query_tokens:
            return []

        scores = []

        if method == 'bm25':
            scores = self.bm25.get_scores(query_tokens)
            
        elif method == 'vsm_cosine':
            q_vec = self.tfidf_cosine_vec.transform([query_proc])
            # cosine_similarity return [[score1, score2...]]
            scores = cosine_similarity(q_vec, self.tfidf_cosine_matrix).flatten()
            
        elif method == 'tfidf_sum':
            q_vec = self.tfidf_sum_vec.transform([query_proc])
            # linear_kernel = dot product (lebih cepat dari cosine karena tanpa normalisasi lagi)
            scores = linear_kernel(q_vec, self.tfidf_sum_matrix).flatten()
            
        elif method == 'jaccard':
            q_set = set(query_tokens)
            # Hitung Jaccard: Intersection / Union
            scores = np.array([
                len(q_set.intersection(d_tok)) / len(q_set.union(d_tok)) if d_tok else 0.0
                for d_tok in self.tokenized_corpus
            ])
        else:
            return []

        # Ambil Top-K Indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            # Filter score 0 agar hasil lebih bersih
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc['score'] = float(scores[idx])
                results.append(doc)
            
        return results