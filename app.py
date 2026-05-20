import os
import sys
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# Setup dynamic pathing to import nlp_config and step4_chatbot_inference
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = r"d:\Capstone\indobert_nlp"

if current_dir not in sys.path:
    sys.path.append(current_dir)

from nlp_config import *
from step4_chatbot_inference import ChatbotSystem

# Initialize FastAPI App
app = FastAPI(
    title="NutriFit AI - IndoBERT Health & Nutrition API",
    description="Backend API serving IndoBERT intent classifier and RAG-based Health Chatbot",
    version="2.0.0"
)

# ============================================================
# CORS Middleware — agar Next.js (atau frontend lain) bisa akses API
# ============================================================
ALLOWED_ORIGINS = [
    "http://localhost:3000",        # Next.js dev server
    "http://localhost:3001",        # Next.js alternate port
    "http://127.0.0.1:3000",
    "https://*.vercel.app",         # Vercel deployments
]

# Jika ada env variable ALLOWED_ORIGINS, gunakan itu (comma-separated)
env_origins = os.environ.get("ALLOWED_ORIGINS", "")
if env_origins:
    ALLOWED_ORIGINS = [o.strip() for o in env_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    print("Initializing Chatbot System (IndoBERT models & datasets)...")
    try:
        chatbot = ChatbotSystem()
        print("Chatbot successfully initialized!")
    except Exception as e:
        print(f"CRITICAL ERROR initializing ChatbotSystem: {e}")
        # Keep running so /health can report the failure and logs can be inspected

# ============================================================
# Request & Response Schemas
# ============================================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    food_extracted: Optional[dict] = None
    response: str

class ClassifyRequest(BaseModel):
    message: str

class ClassifyResponse(BaseModel):
    intent: str
    confidence: Optional[float] = None

class FoodSearchResponse(BaseModel):
    results: list
    total: int

class FoodDetailResponse(BaseModel):
    found: bool
    data: Optional[dict] = None

class ExerciseResponse(BaseModel):
    results: list
    total: int

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def redirect_to_docs():
    """Redirects base URL to Swagger interactive API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health")
def get_health():
    """Liveness & Readiness probe for Google Cloud Run health checks"""
    if chatbot is None:
        return {
            "status": "unhealthy", 
            "message": "Chatbot model failed to load. Check server logs."
        }
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_llm": MODEL_LLM,
        "cuda_available": chatbot.model.device.type == "cuda"
    }


# --- 1. MAIN CHAT ENDPOINT (untuk Next.js chatbot page) ---
@app.post("/api/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """
    Endpoint utama chatbot.
    Menerima pesan user, jalankan IndoBERT intent classification,
    ekstrak data makanan via TF-IDF, dan generate response via Nvidia NIM LLM.
    
    Dipanggil dari Next.js: POST /api/chat { "message": "berapa kalori nasi goreng?" }
    """
    if chatbot is None:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot system is not loaded or failed to initialize."
        )
    
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")
    
    try:
        # 1. Run IndoBERT Intent Classifier
        intent = chatbot.get_intent(req.message)
        
        # 2. Extract food using TF-IDF cosine similarity
        food_data = chatbot.extract_food(req.message)
        food_dict = None
        
        if food_data is not None:
            # Convert pandas Row to dictionary, replace NaN with empty strings
            food_dict = food_data.fillna("").to_dict()
            # Clean up keys and numbers
            for key, val in food_dict.items():
                if isinstance(val, float) and pd.isna(val):
                    food_dict[key] = ""
        
        # 3. Generate response using the main system
        response = chatbot.generate_response(req.message)
        
        return ChatResponse(
            intent=intent,
            food_extracted=food_dict,
            response=response
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- 2. CLASSIFY ONLY (intent tanpa generate response — cepat, ringan) ---
@app.post("/api/classify", response_model=ClassifyResponse)
def post_classify(req: ClassifyRequest):
    """
    Klasifikasi intent saja tanpa memanggil LLM.
    Cocok untuk Next.js yang cuma perlu tahu intent user sebelum routing logic.
    
    Dipanggil dari Next.js: POST /api/classify { "message": "carikan resep ayam bakar" }
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")
    
    try:
        intent = chatbot.get_intent(req.message)
        return ClassifyResponse(intent=intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 3. FOOD SEARCH (cari makanan di knowledge base) ---
@app.get("/api/food/search", response_model=FoodSearchResponse)
def search_food(
    q: str = Query(..., description="Kata kunci pencarian makanan"),
    limit: int = Query(10, ge=1, le=50, description="Jumlah hasil maksimal")
):
    """
    Search makanan di Knowledge Base menggunakan TF-IDF similarity.
    
    Dipanggil dari Next.js: GET /api/food/search?q=nasi+goreng&limit=5
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vec = chatbot.vectorizer.transform([q])
        similarities = cosine_similarity(query_vec, chatbot.tfidf_matrix)[0]
        
        # Ambil top-N results yang similarity > 0
        top_indices = similarities.argsort()[::-1][:limit]
        results = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                row = chatbot.kb.iloc[idx].fillna("").to_dict()
                row["similarity_score"] = round(score, 4)
                results.append(row)
        
        return FoodSearchResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 4. FOOD DETAIL (ambil detail satu makanan by nama) ---
@app.get("/api/food/{food_name}", response_model=FoodDetailResponse)
def get_food_detail(food_name: str):
    """
    Ambil detail nutrisi satu makanan dari Knowledge Base.
    
    Dipanggil dari Next.js: GET /api/food/nasi%20goreng
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        matches = chatbot.kb[
            chatbot.kb['nama_makanan'].str.lower().str.contains(food_name.lower(), na=False)
        ]
        
        if matches.empty:
            return FoodDetailResponse(found=False)
        
        # Ambil match pertama
        row = matches.iloc[0].fillna("").to_dict()
        return FoodDetailResponse(found=True, data=row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. EXERCISES LIST (data kalori olahraga) ---
@app.get("/api/exercises", response_model=ExerciseResponse)
def get_exercises(
    q: Optional[str] = Query(None, description="Filter nama olahraga (opsional)"),
    limit: int = Query(20, ge=1, le=100, description="Jumlah hasil")
):
    """
    Ambil daftar olahraga beserta kalori yang terbakar.
    
    Dipanggil dari Next.js: GET /api/exercises?q=lari&limit=10
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if chatbot.exercise_db is None:
        return ExerciseResponse(results=[], total=0)
    
    try:
        df = chatbot.exercise_db.copy()
        
        if q:
            # Filter berdasarkan query jika ada
            mask = df.apply(
                lambda row: row.astype(str).str.lower().str.contains(q.lower()).any(), axis=1
            )
            df = df[mask]
        
        df = df.head(limit)
        results = df.fillna("").to_dict(orient="records")
        
        return ExerciseResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Cloud Run passes the port as an environment variable PORT
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting API server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
