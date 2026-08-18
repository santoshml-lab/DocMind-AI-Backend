from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import io
import os

from huggingface_hub import InferenceClient
from supabase import create_client
from groq import Groq


app = FastAPI(
    title="DocMind AI",
    description="AI-powered document knowledge assistant using RAG",
    version="1.0.0"
)


# =========================================================
# CONFIGURATION
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
GROQ_MODEL = "openai/gpt-oss-20b"


if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not configured.")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is not configured.")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is not configured.")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")


# Hugging Face client
hf_client = InferenceClient(
    token=HF_TOKEN
)


# Supabase client
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# Groq client
groq_client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "DocMind AI Backend is running 🚀"
    }


# =========================================================
# CHUNKING
# =========================================================

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


# =========================================================
# EMBEDDING
# =========================================================

def generate_embedding(text, prefix="passage"):

    try:

        embedding = hf_client.feature_extraction(
            f"{prefix}: {text}",
            model=EMBEDDING_MODEL
        )

        # Convert numpy array / tensor-like result to list
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # Some responses can contain an extra dimension
        if embedding and isinstance(embedding[0], list):
            embedding = embedding[0]

        return embedding

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {str(e)}"
        )


# =========================================================
# UPLOAD PDF
# =========================================================

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


    # -----------------------------------------------------
    # PDF TEXT EXTRACTION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CREATE CHUNKS
    # -----------------------------------------------------

    chunks = chunk_text(extracted_text)


    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="No readable text found in the PDF."
        )


    # -----------------------------------------------------
    # STORE CHUNKS + EMBEDDINGS
    # -----------------------------------------------------

    stored_chunks = 0


    try:

        for index, chunk in enumerate(chunks):

            embedding = generate_embedding(
                chunk,
                prefix="passage"
            )


            # Verify embedding dimension
            if len(embedding) != 1024:

                raise ValueError(
                    f"Expected 1024 dimensions, got {len(embedding)}"
                )


            supabase.table("document_chunks").insert({

                "filename": file.filename,

                "chunk_index": index,

                "content": chunk,

                "embedding": embedding

            }).execute()


            stored_chunks += 1


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to store document chunks: {str(e)}"
        )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "message": "PDF processed and stored successfully 🚀",

        "filename": file.filename,

        "pages": page_count,

        "chunks_count": len(chunks),

        "stored_chunks": stored_chunks

    }


# =========================================================
# EMBEDDING TEST
# =========================================================

@app.post("/embed-test")
async def embed_test(data: dict):

    text = data.get("text")


    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text is required."
        )


    embedding = generate_embedding(
        text,
        prefix="passage"
    )


    return {

        "text": text,

        "dimensions": len(embedding),

        "embedding": embedding

    }


# =========================================================
# VECTOR SEARCH
# =========================================================

@app.post("/search")
async def search_documents(data: dict):

    query = data.get("query")

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query is required."
        )


    # -----------------------------------------------------
    # CREATE QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = generate_embedding(
        query,
        prefix="query"
    )


    if len(query_embedding) != 1024:

        raise HTTPException(
            status_code=500,
            detail=f"Expected 1024 dimensions, got {len(query_embedding)}"
        )


    # -----------------------------------------------------
    # VECTOR SIMILARITY SEARCH
    # -----------------------------------------------------

    try:

        response = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": 5
            }
        ).execute()


        results = response.data


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "query": query,

        "results": results

    }


# =========================================================
# RAG ASK
# =========================================================

@app.post("/ask")
async def ask_question(data: dict):

    query = data.get("query")


    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query is required."
        )


    # -----------------------------------------------------
    # CREATE QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = generate_embedding(
        query,
        prefix="query"
    )


    if len(query_embedding) != 1024:

        raise HTTPException(
            status_code=500,
            detail=f"Expected 1024 dimensions, got {len(query_embedding)}"
        )


    # -----------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENT CHUNKS
    # -----------------------------------------------------

    try:

        response = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": 5
            }
        ).execute()


        results = response.data


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )


    if not results:

        return {
            "query": query,
            "answer": "I could not find relevant information in the uploaded documents."
        }


    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context_parts = []

    for result in results:

        content = result.get("content")

        if content:

            context_parts.append(content)


    context = "\n\n--- DOCUMENT CHUNK ---\n\n".join(
        context_parts
    )


    # -----------------------------------------------------
    # GROQ GENERATION
    # -----------------------------------------------------

    prompt = f"""
You are DocMind AI, a document knowledge assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not present in the context, say:
"I could not find that information in the uploaded document."

Do not invent facts.
Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}
"""


    try:

        completion = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": "You answer questions using provided document context only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=500

        )


        answer = completion.choices[0].message.content


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Groq generation failed: {str(e)}"
        )


    # -----------------------------------------------------
    # FINAL RAG RESPONSE
    # -----------------------------------------------------

    return {

        "query": query,

        "answer": answer,

        "sources": [
            {
                "filename": result.get("filename"),
                "chunk_index": result.get("chunk_index"),
                "similarity": result.get("similarity")
            }
            for result in results
        ]

    }

    


    

        
        
    
