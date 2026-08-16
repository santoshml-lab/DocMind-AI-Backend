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

    return {
        "filename": file.filename,
        "pages": page_count,
        "text": extracted_text
    }
