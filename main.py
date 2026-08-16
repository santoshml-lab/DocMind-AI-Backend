from fastapi import FastAPI

app = FastAPI(
    title="DocMind AI",
    description="AI-powered document knowledge assistant using RAG",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "DocMind AI Backend is running 🚀"
    }
