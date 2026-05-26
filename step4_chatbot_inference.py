import torch
import sys, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys, os
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = r"d:\Capstone\indobert_nlp"
sys.path.append(current_dir)
from nlp_config import *

from openai import OpenAI

class ChatbotSystem:
    def __init__(self):
        print("Memuat Knowledge Base Makanan...")
        self.kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
        self.kb['nama_makanan'] = self.kb['nama_makanan'].fillna('')
        
        print("Memuat Data Kalori Olahraga...")
        if EXERCISE_DB_PATH.exists():
            self.exercise_db = pd.read_csv(EXERCISE_DB_PATH)
        else:
            self.exercise_db = None
            
        print("Membuat Retriever Engine (TF-IDF)...")
        # Daftar kata umum bahasa Indonesia yang harus diabaikan saat mencari nama makanan
        INDONESIAN_STOPWORDS = [
            "saya", "kamu", "dia", "mereka", "kita", "kami", "anda", "aku",
            "yang", "di", "ke", "dari", "dan", "atau", "dengan", "untuk", "pada",
            "adalah", "itu", "ini", "buat", "dong", "ya", "sih", "berapa", "kalori",
            "tinggi", "berat", "umur", "tahun", "tubuh", "badan", "hitung",
            "makanan", "resep", "menu", "sehat", "porsi", "siang", "makan"
        ]
        self.vectorizer = TfidfVectorizer(stop_words=INDONESIAN_STOPWORDS)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.kb['nama_makanan'])
        
        print("Memuat Model IndoBERT...")
        # Coba load dari folder lokal dulu (biar cepet saat testing lokal)
        local_model_path = MODEL_OUTPUT_DIR
        local_safetensors = local_model_path / "model.safetensors"
        
        if local_model_path.exists() and local_safetensors.exists():
            print(f"✅ Memuat model dari folder lokal: {local_model_path}")
            model_source = str(local_model_path)
        else:
            # Jika lokal tidak ada (seperti di server Railway), download dari Hugging Face
            hf_repo = os.environ.get("HF_MODEL_REPO", "s4usan/indobert-intent-classifier")
            print(f"☁️ Memuat model dari Hugging Face Hub: {hf_repo}")
            model_source = hf_repo
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
        
        print(f"Menginisialisasi LLM RAG Engine ({ACTIVE_LLM_URL}) untuk model: {ACTIVE_LLM_MODEL}...")
        self.llm_client = OpenAI(
            base_url=ACTIVE_LLM_URL,
            api_key=ACTIVE_LLM_KEY.strip(),
            timeout=20.0
        )
        
    def get_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        
        intent = self.model.config.id2label[predicted_class_id]
        return intent
        
    def extract_food(self, query):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = similarities.argmax()
        
        if similarities[best_idx] > 0.5:
            return self.kb.iloc[best_idx]
        return None

    def ask_llm(self, prompt):
        # Masukkan model utama dan alternatif Groq ke list
        models_to_try = [ACTIVE_LLM_MODEL, "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
        provider_name = "Groq Cloud"
            
        unique_models = []
        for m in models_to_try:
            if m not in unique_models:
                unique_models.append(m)
        
        last_error = None
        for model in unique_models:
            try:
                completion = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Anda adalah asisten chatbot kesehatan dan nutrisi yang ramah."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1024,
                    stream=False,
                    timeout=30.0
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[Warning] Gagal menghubungi model {model} via {provider_name}: {e}. Mencoba model alternatif...")
                last_error = e
        
        return f"Maaf, sedang ada kendala koneksi ke {provider_name}. {str(last_error)}"

    def generate_response(self, user_input):
        intent = self.get_intent(user_input)
        
        # 1. Intent Salam/Terima Kasih (Langsung LLM)
        if intent in ['salam', 'terima_kasih']:
            prompt = f"User mengatakan: '{user_input}'. Balas dengan ramah dan tawarkan bantuan tentang nutrisi makanan atau resep."
            return self.ask_llm(prompt)
            
        # 2. Intent Olahraga
        if intent == "cari_kalori_olahraga":
            context = "Berikut beberapa data pembakaran kalori per jam: "
            if self.exercise_db is not None:
                context += self.exercise_db.head(10).to_string()
            
            prompt = f"User bertanya tentang kalori olahraga: '{user_input}'. \nKonteks Data: {context}\nJawablah dengan informatif dan berikan tips olahraga."
            return self.ask_llm(prompt)

        # 3. Intent Makanan (Kalori/Nutrisi/Resep)
        food_data = self.extract_food(user_input)
        
        if food_data is None:
            # Pengecekan apakah user bertanya tentang kebutuhan kalori tubuh/BMR/TDEE sendiri
            personal_keywords = ["tubuh saya", "kebutuhan saya", "berat", "tinggi", "umur", "usia", "bmr", "tdee", "kalori harian", "kebutuhan kalori"]
            if any(kw in user_input.lower() for kw in personal_keywords):
                prompt = (
                    f"User bertanya tentang kebutuhan kalori tubuhnya / BMR: '{user_input}'. "
                    f"Jawablah dengan ramah, bantu hitung BMR (Basal Metabolic Rate) dan TDEE (Total Daily Energy Expenditure) "
                    f"jika ada info berat, tinggi, usia, atau jenis kelamin di dalam pertanyaan. Berikan penjelasan kesehatan yang informatif."
                )
                return self.ask_llm(prompt)
            
            prompt = (
                f"User bertanya tentang topik kesehatan/nutrisi: '{user_input}'. "
                f"Meskipun data spesifik tidak ditemukan di database menu lokal kami, "
                f"jawablah pertanyaan tersebut menggunakan pengetahuan umum Anda tentang kesehatan dan nutrisi "
                f"secara informatif, ramah, dan praktis."
            )
            return self.ask_llm(prompt)
        
        nama = food_data['nama_makanan'].title()
        
        # Siapkan Konteks Data dari Database
        data_context = f"Nama Makanan: {nama}\n"
        if 'kalori' in food_data: data_context += f"Kalori: {food_data['kalori']} kkal\n"
        if 'protein_g' in food_data: data_context += f"Protein: {food_data['protein_g']}g\n"
        if 'bahan' in food_data: data_context += f"Bahan/Resep: {food_data['bahan']}\n"
        if 'langkah_resep' in food_data: data_context += f"Langkah: {food_data['langkah_resep']}\n"

        # Buat Prompt RAG
        if intent == "cari_kalori":
            prompt = f"User bertanya kalori: '{user_input}'. \nData dari database:\n{data_context}\nJelaskan jumlah kalori ini ke user dengan gaya yang ramah dan beritahu apakah ini sehat atau tidak."
        elif intent == "cari_nutrisi":
            prompt = f"User bertanya nutrisi: '{user_input}'. \nData dari database:\n{data_context}\nJelaskan detail nutrisi ini (protein, lemak, karbo) secara lengkap dan informatif."
        elif intent == "cari_resep":
            prompt = f"User meminta resep: '{user_input}'. \nData dari database:\n{data_context}\nBerikan resep lengkap beserta tips memasaknya agar lebih sehat."
        else:
            prompt = f"User bertanya: '{user_input}'. \nData relevan: {data_context}\nBerikan jawaban yang sesuai konteks."

        return self.ask_llm(prompt)

def main():
    if not MODEL_OUTPUT_DIR.exists():
        print("Model belum ditraining! Silakan run step 3 terlebih dahulu.")
        return
        
    chatbot = ChatbotSystem()
    print("\n" + "="*50)
    print("🤖 Chatbot Nutrisi Siap!")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Anda: ")
        if user_input.lower() in ['keluar', 'exit', 'quit']:
            break
            
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
