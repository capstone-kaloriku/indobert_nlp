import os
import sys
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
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
    version="1.0.0"
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

# Request and Response schemas
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    food_extracted: dict = None
    response: str

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

@app.post("/api/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """Processes user query, runs IndoBERT intent classification, retrieves data, and requests Nvidia NIM response"""
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

if __name__ == "__main__":
    # Cloud Run passes the port as an environment variable PORT
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting API server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
