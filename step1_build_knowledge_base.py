import pandas as pd
import numpy as np
import sys, os
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = r"d:\Capstone\indobert_nlp"
sys.path.append(current_dir)
from nlp_config import *
try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("WARNING: deep-translator belum diinstall. Run 'pip install deep-translator' dulu.")
    import sys; sys.exit(1)

def build_knowledge_base():
    print("Memulai pembuatan Knowledge Base...")
    
    dfs = []
    
    # 1. Dataset 1
    if DS1_PATH.exists():
        print("Memproses Dataset 1...")
        df1 = pd.read_csv(DS1_PATH)
        df_std1 = pd.DataFrame({
            'nama_makanan': df1['Title'],
            'jenis_makanan': df1['jenis_makanan'],
            'kalori': df1['jumlah_kalori'],
            'bahan': df1['Ingredients'],
            'langkah_resep': df1['Steps'],
            'usia': df1['usia'].astype(str),
            'sumber': 'DS1'
        })
        dfs.append(df_std1)

    # 2. Dataset 2 (Data Olahraga)
    if DS2_PATH.exists():
        print("Memproses Dataset 2 (Data Olahraga)...")
        df2 = pd.read_csv(DS2_PATH)
        df2.to_csv(EXERCISE_DB_PATH, index=False)
        print(f"-> Data olahraga disimpan terpisah ke {EXERCISE_DB_PATH}")

    # 3. Dataset 3
    if DS3_PATH.exists():
        print("Memproses Dataset 3...")
        df3 = pd.read_csv(DS3_PATH)
        df_std3 = pd.DataFrame({
            'nama_makanan': df3['name'],
            'kalori': df3['calories'],
            'protein_g': df3['proteins'],
            'lemak_g': df3['fat'],
            'karbohidrat_g': df3['carbohydrate'],
            'usia': df3['age_group'],
            'sumber': 'DS3'
        })
        dfs.append(df_std3)
        
    # 4. Dataset 4 (English Food per 100g -> Translated)
    if DS4_PATH.exists():
        print("Memproses Dataset 4 (Menerjemahkan nama makanan dari Inggris ke Indonesia)...")
        print("Mohon tunggu, proses translate mungkin memakan waktu beberapa menit...")
        df4 = pd.read_csv(DS4_PATH)
        translator = GoogleTranslator(source='en', target='id')
        
        translated_names = []
        # Untuk menghemat waktu, kita cuma ambil 500 data pertama atau keseluruhan (karena dataset bisa besar)
        # Jika ukuran data DS4 kecil (misal ratusan baris), kita translate semua.
        total_rows = min(len(df4), 500) 
        df4_subset = df4.head(total_rows).copy()
        
        for i, name in enumerate(df4_subset['food_normalized'].fillna('')):
            if name:
                try:
                    res = translator.translate(name)
                    translated_names.append(res)
                except Exception as e:
                    translated_names.append(name)
            else:
                translated_names.append('')
            
            # Progress bar simple
            if (i+1) % 50 == 0:
                print(f"  Translated {i+1}/{total_rows} items...")
                
        df_std4 = pd.DataFrame({
            'nama_makanan': translated_names,
            'kalori': df4_subset['Calories (kcal per 100g)'],
            'protein_g': df4_subset['Protein (g per 100g)'],
            'lemak_g': df4_subset['Fat (g per 100g)'],
            'karbohidrat_g': df4_subset['Carbohydrates (g per 100g)'],
            'natrium_mg': df4_subset['Sodium (mg per 100g)'],
            'gula_g': df4_subset['Sugars (g per 100g)'],
            'serat_g': df4_subset['Dietary Fiber (g per 100g)'],
            'sumber': 'DS4'
        })
        dfs.append(df_std4)

    # 5. Dataset 5
    if DS5_PATH.exists():
        print("Memproses Dataset 5...")
        df5 = pd.read_csv(DS5_PATH)
        df_std5 = pd.DataFrame({
            'nama_makanan': df5['Title_cleaned'],
            'jenis_makanan': df5['Food Type'],
            'bahan': df5['Ingredients_cleaned'],
            'langkah_resep': df5['Steps_cleaned'],
            'sumber': 'DS5'
        })
        dfs.append(df_std5)

    # 6. Dataset 6
    if DS6_PATH.exists():
        print("Memproses Dataset 6...")
        df6 = pd.read_csv(DS6_PATH)
        df_std6 = pd.DataFrame({
            'nama_makanan': df6['jenis_makanan'],
            'kalori': df6['kalori'],
            'protein_g': df6['protein_g'],
            'karbohidrat_g': df6['carbohydrate_g'],
            'lemak_g': df6['fat_g'],
            'gula_g': df6['sugar_g'],
            'natrium_mg': df6['sodium_mg'],
            'serat_g': df6['fiber_g'],
            'usia': df6['usia'].astype(str),
            'sumber': 'DS6'
        })
        dfs.append(df_std6)

    print("Menggabungkan dataset makanan...")
    kb = pd.concat(dfs, ignore_index=True)
    
    # Cleaning
    kb['nama_makanan'] = kb['nama_makanan'].str.lower().str.strip()
    kb = kb.drop_duplicates(subset=['nama_makanan'], keep='first')
    
    kb.to_csv(KNOWLEDGE_BASE_PATH, index=False)
    print(f"Knowledge Base Makanan berhasil disimpan ke {KNOWLEDGE_BASE_PATH}")
    print(f"Total data makanan: {len(kb)}")

if __name__ == "__main__":
    build_knowledge_base()
