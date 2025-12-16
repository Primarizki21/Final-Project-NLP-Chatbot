import pandas as pd
import time
import os
import numpy as np
from rag_utils_baru import RetrievalSystem

# --- KONFIGURASI ---
DATA_FILE = "data_baru.xlsx"
GT_FILE = "new_ground_truth_baru.xlsx"
OUTPUT_FILE = "hasil_evaluasi_lengkap.xlsx" # Ganti nama file biar fresh
METHODS = ['bm25', 'tfidf_sum', 'jaccard', 'vsm_cosine']
K_VALUES = [1, 3, 5, 10]

def parse_target_ids(raw_val):
    str_val = str(raw_val).strip()
    if '.' in str_val and ',' not in str_val:
        str_val = str_val.replace('.', ',')

    # Hapus suffix .0 standar (misal 4.0 -> 4)
    if str_val.endswith('.0'):
        str_val = str_val[:-2]

    parts = str_val.split(',')
    
    clean_ints = []
    for p in parts:
        p = p.strip()
        # Bersihkan lagi jika ada sisa .0 di setiap bagian
        if p.endswith('.0'):
            p = p[:-2]
            
        if p: # Jika tidak kosong
            try:
                # KONVERSI KE INTEGER
                val_int = str(p)
                clean_ints.append(val_int)
            except ValueError:
                pass 
                
    return clean_ints

def main():
    # 1. Cek Path File Output (Untuk debugging file tidak ketemu)
    abs_output_path = os.path.abspath(OUTPUT_FILE)
    print(f"[INFO] File output nanti akan disimpan di:\n -> {abs_output_path}\n")

    # 2. Inisialisasi Engine
    try:
        engine = RetrievalSystem(DATA_FILE)
    except Exception as e:
        print(f"[STOP] Error init: {e}")
        return

    # 3. Load Ground Truth
    print(f"[EVAL] Membaca GT dari {GT_FILE}...")
    try:
        df_gt = pd.read_excel(GT_FILE, sheet_name="GT", dtype=str)
    except Exception as e:
        print(f"[STOP] Gagal baca GT: {e}")
        return
        
    all_results = []
    total_data = len(df_gt)

    # 4. Loop Evaluasi
    for i, row in df_gt.iterrows():
        query = str(row['question'])
        target_type = str(row['source_type']).strip()
        target_ids_list = parse_target_ids(row['source_id'])
        
        # Jumlah dokumen relevan yang tersedia (untuk penyebut Recall)
        total_relevant = len(target_ids_list) 
        if total_relevant == 0: continue # Skip jika tidak ada kunci jawaban

        print(f"\n[{i+1}/{total_data}] Q: {query[:40]}...")
        print(f"   -> TARGET: {target_ids_list} (Type: {target_type})")

        for method in METHODS:
            start_time = time.time()
            
            # Retrieve max K
            retrieved = engine.search(query, method, top_k=max(K_VALUES))
            duration = time.time() - start_time
            
            # --- ANALISIS HASIL RETRIEVAL ---
            retrieved_ids_debug = []  # Untuk laporan excel
            matches_vector = []       # [1, 0, 1, ...]
            
            for doc in retrieved:
                doc_id = str(doc['source_id'])
                doc_type = str(doc['source_type'])
                
                # Simpan info dokumen yang terambil
                retrieved_ids_debug.append(f"{doc_id}")
                
                # Cek Relevansi (ID ada di target DAN Tipe Sheet sama)
                is_match = (doc_type == target_type) and (doc_id in target_ids_list)
                matches_vector.append(1 if is_match else 0)
            
            # Print debug singkat ke layar (hanya top 5 biar ga spam)
            top_retrieved_str = ", ".join(retrieved_ids_debug[:5])
            print(f"   [{method}] Found: [{top_retrieved_str}...] | Matches: {sum(matches_vector)}")

            # --- HITUNG METRICS (Precision & Recall) ---
            res_row = {
                'No': i+1,
                'Question': query,
                'Target Type': target_type,
                'Total Relevant (Ground Truth)': total_relevant,
                'Target IDs': target_ids_list,
                'Method': method,
                'Retrieved IDs (Top-K)': ", ".join(retrieved_ids_debug), # Ini yang kamu minta
                'Time': duration
            }
            
            for k in K_VALUES:
                # Ambil irisan k pertama
                current_matches = matches_vector[:k]
                relevant_retrieved = sum(current_matches) # Jumlah dokumen relevan yang ditemukan di top K
                
                # Precision@K: (Relevan di Top K) / K
                precision = relevant_retrieved / k
                
                # Recall@K: (Relevan di Top K) / (Total Dokumen Relevan di Database)
                recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
                
                res_row[f'Precision@{k}'] = precision
                res_row[f'Recall@{k}'] = recall
            
            all_results.append(res_row)

    # 5. Simpan Hasil
    print("\n[EVAL] Menyimpan laporan...")
    if not all_results:
        print("[ERROR] Tidak ada hasil yang diproses. Cek nama sheet/kolom di excel.")
        return

    df_res = pd.DataFrame(all_results)
    
    # Buat Summary Rata-rata
    metric_cols = [f'Precision@{k}' for k in K_VALUES] + [f'Recall@{k}' for k in K_VALUES]
    summary = df_res.groupby('Method')[metric_cols].mean()
    
    try:
        with pd.ExcelWriter(OUTPUT_FILE, mode='w') as writer:
            df_res.to_excel(writer, sheet_name='Detail', index=False)
            summary.to_excel(writer, sheet_name='Summary')
        print(f"\n[SUKSES] File tersimpan di: {abs_output_path}")
        print("Silakan buka file tersebut untuk melihat detail Retrieved ID.")
        
    except PermissionError:
        print(f"\n[ERROR] File '{OUTPUT_FILE}' sedang dibuka di Excel!")
        print("Tutup file Excel tersebut lalu jalankan ulang script ini.")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan file: {e}")

    print("\n=== HASIL RATA-RATA ===")
    print(summary)

if __name__ == "__main__":
    main()