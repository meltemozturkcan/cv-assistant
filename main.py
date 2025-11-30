from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chatbot import initialize_chatbot, ask_question
from fastapi.middleware.cors import CORSMiddleware

# FastAPI uygulaması
app = FastAPI(
    title="Meltem Öztürkcan CV Chatbot API",
    description="RAG tabanlı CV soru-cevap API'si",
    version="1.0.0"
)
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://cv-assistant-ui.onrender.com",  # ✅ canlı frontend domaini
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Chatbot'u başlat
chatbot = None

@app.on_event("startup")
async def startup_event():
    global chatbot
    print("🚀 Chatbot başlatılıyor...")
    chatbot = initialize_chatbot()
    print("✅ Chatbot hazır!")

# Request/Response modelleri
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# Endpoints
@app.get("/")
async def root():
    return {"message": "CV Chatbot API çalışıyor!", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chatbot_ready": chatbot is not None}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot henüz hazır değil")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz")
    
    answer = ask_question(chatbot, request.question)
    
    return AnswerResponse(
        question=request.question,
        answer=answer
    )

# Çalıştırma
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
