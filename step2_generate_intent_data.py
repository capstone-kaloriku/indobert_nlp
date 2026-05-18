import pandas as pd
import random
import sys, os
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = r"d:\Capstone\indobert_nlp"
sys.path.append(current_dir)
from nlp_config import *

INTENT_TEMPLATES = {
    "cari_kalori": [
        "berapa kalori {makanan}",
        "kalori {makanan} berapa",
        "{makanan} ada berapa kalorinya",
        "mau tahu kalori {makanan}",
        "info kalori {makanan} dong",
        "hitung kalori {makanan}",
        "kandungan kalori {makanan}",
        "cek kalori {makanan}",
        "apakah {makanan} kalorinya tinggi"
    ],
    "cari_nutrisi": [
        "apa kandungan gizi {makanan}",
        "nutrisi {makanan} apa saja",
        "info nutrisi {makanan}",
        "kandungan protein {makanan} berapa",
        "cek gizi {makanan}",
        "berapa protein dan lemak di {makanan}",
        "detail nutrisi untuk {makanan}"
    ],
    "cari_resep": [
        "bagaimana cara membuat {makanan}",
        "resep {makanan} dong",
        "cara masak {makanan}",
        "minta resep {makanan}",
        "bahan untuk membuat {makanan}",
        "langkah memasak {makanan}",
        "ajari aku bikin {makanan}"
    ]
}

GENERAL_INTENTS = {
    "salam": [
        "halo", "hai", "selamat pagi", "selamat siang", "selamat malam", 
        "hei", "halo bot", "permisi"
    ],
    "terima_kasih": [
        "terima kasih", "makasih", "thanks", "makasih ya", "ok makasih",
        "terima kasih banyak"
    ],
    "cari_kalori_olahraga": [
        "kalau olahraga bakar berapa kalori",
        "berapa kalori terbakar saat olahraga",
        "hitung kalori olahraga",
        "kalori olahraga dong",
        "olahraga membakar berapa kalori",
        "berapa pembakaran kalori saat senam atau lari",
        "info kalori olahraga",
        "kalori aktivitas fisik",
        "kalkulator kalori olahraga"
    ]
}

def generate_intent_data():
    print("Memulai pembuatan data training intent...")
    
    if not KNOWLEDGE_BASE_PATH.exists():
        print("Knowledge base belum dibuat! Silakan run step 1 dulu.")
        return
        
    kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
    makanan_list = kb['nama_makanan'].dropna().tolist()
    
    if len(makanan_list) > 1000:
        makanan_list = random.sample(makanan_list, 1000)
        
    data = []
    
    for makanan in makanan_list:
        makanan_bersih = str(makanan).strip()
        if not makanan_bersih: continue
            
        for intent, templates in INTENT_TEMPLATES.items():
            for template in templates:
                if random.random() < 0.3:
                    text = template.format(makanan=makanan_bersih)
                    data.append({"text": text, "intent": intent})
                    
    for intent, texts in GENERAL_INTENTS.items():
        for _ in range(600):
            for text in texts:
                variasi = text + (" ya" if random.random() > 0.5 else "")
                data.append({"text": variasi, "intent": intent})
                
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(INTENT_DATA_PATH, index=False)
    print(f"Data intent berhasil disimpan ke {INTENT_DATA_PATH}")
    print(f"Total baris data training: {len(df)}")
    print("\nDistribusi Intent:")
    print(df['intent'].value_counts())

if __name__ == "__main__":
    generate_intent_data()
