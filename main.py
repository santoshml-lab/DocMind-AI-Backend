from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import io

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

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_bytes = await file.read()

    extracted_text = ""
    page_count = 0

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:

            page_count = len(pdf.pages)

            for page in pdf.pages:
                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

chunks = chunk_text(extracted_text)

   return {
    "filename": file.filename,
    "pages": page_count,
    "chunks_count": len(chunks),
    "chunks": chunks
        }

    


    

        
        
    
