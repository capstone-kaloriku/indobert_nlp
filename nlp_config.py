import os
from pathlib import Path

# Mendapatkan lokasi direktori tempat file nlp_config.py berada secara dinamis
NLP_DIR = Path(__file__).parent.resolve()

# PROJECT_ROOT berada satu level di atas NLP_DIR (yaitu folder Capstone di local)
PROJECT_ROOT = NLP_DIR.parent.resolve()

DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = NLP_DIR / "data"
MODELS_DIR = NLP_DIR / "models"

# Buat direktori jika belum ada
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Path ke masing-masing dataset
DS1_PATH = DATASET_DIR / "Dataset 1" / "cleaned_nutrition_data" / "cleaned_nutrition_data.csv"
DS2_PATH = DATASET_DIR / "Dataset 2" / "cleaned_calories" / "cleaned_calories.csv"
DS3_PATH = DATASET_DIR / "Dataset 3" / "nutrition_cleaned" / "nutrition_cleaned.csv"
DS4_PATH = DATASET_DIR / "Dataset 4" / "cleaned_nutrition_dataset_per100g" / "cleaned_nutrition_dataset_per100g.csv"
DS5_PATH = DATASET_DIR / "Dataset 5" / "kaloriku_processed_recipes" / "kaloriku_processed_recipes.csv"
DS6_PATH = DATASET_DIR / "Dataset 6" / "nilai_gizi_cleaned" / "nilai_gizi_cleaned.csv"

# Output paths
KNOWLEDGE_BASE_PATH = DATA_DIR / "unified_knowledge_base.csv"
EXERCISE_DB_PATH = DATA_DIR / "exercise_calorie_data.csv"
INTENT_DATA_PATH = DATA_DIR / "intent_training_data.csv"
MODEL_OUTPUT_DIR = MODELS_DIR / "indobert_intent_classifier"

INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"

# --- LLM SETTINGS (NVIDIA NIM) ---
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not NVIDIA_API_KEY:
    print("⚠️  WARNING: NVIDIA_API_KEY environment variable not set. LLM features will fail.")
    print("   Set it with: export NVIDIA_API_KEY=nvapi-xxxxx  (Linux/Mac)")
    print("   Or:          $env:NVIDIA_API_KEY='nvapi-xxxxx'  (PowerShell)")
MODEL_LLM = "meta/llama-3.3-70b-instruct"

# --- DEPRECATED GEMINI SETTINGS ---
# GEMINI_API_KEY = "AIzaSyAkiv37rewhUmA_lcRsbyBMWK40tEZu4Rc"
# MODEL_GEMINI = "gemini-2.5-pro"