from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import io
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware

from huggingface_hub import InferenceClient
from supabase import create_client
from groq import Groq


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="DocMind AI",
    description="AI-powered document knowledge assistant using RAG",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://doc-mind-ai-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# =========================================================
# ENVIRONMENT CHECK
# =========================================================

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not configured.")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is not configured.")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is not configured.")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")


# =========================================================
# CLIENTS
# =========================================================

hf_client = InferenceClient(
    token=HF_TOKEN
)


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


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

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):

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

def generate_embedding(
    text,
    prefix="passage"
):

    try:

        embedding = hf_client.feature_extraction(
            f"{prefix}: {text}",
            model=EMBEDDING_MODEL
        )


        # Convert numpy / tensor-like object
        # into normal Python list
        if hasattr(embedding, "tolist"):

            embedding = embedding.tolist()


        # Remove extra dimension if returned
        # as [[...]]
        if (
            embedding
            and isinstance(embedding[0], list)
        ):

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
async def upload_pdf(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # FILE VALIDATION
    # -----------------------------------------------------

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )


    file_bytes = await file.read()
    document_id = str(uuid.uuid4())

    extracted_text = ""

    page_count = 0


    # -----------------------------------------------------
    # PDF TEXT EXTRACTION
    # -----------------------------------------------------

    try:

        with pdfplumber.open(
            io.BytesIO(file_bytes)
        ) as pdf:

            page_count = len(pdf.pages)


            for page in pdf.pages:

                text = page.extract_text()


                if text:

                    extracted_text += (
                        text + "\n"
                    )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )


    # -----------------------------------------------------
    # CREATE CHUNKS
    # -----------------------------------------------------

    chunks = chunk_text(
        extracted_text
    )


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


            # Verify dimension
            if len(embedding) != 1024:

                raise ValueError(
                    f"Expected 1024 dimensions, "
                    f"got {len(embedding)}"
                )


            supabase.table(
                "document_chunks"
            ).insert({
                "document_id": document_id,

                "filename": file.filename,

                "chunk_index": index,

                "content": chunk,

                "embedding": embedding

            }).execute()


            stored_chunks += 1


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to store document chunks: "
                f"{str(e)}"
            )
        )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "message":
            "PDF processed and stored successfully 🚀",
        "document_id":
            document_id,

        "filename":
            file.filename,

        "pages":
            page_count,

        "chunks_count":
            len(chunks),

        "stored_chunks":
            stored_chunks

    }


# =========================================================
# EMBEDDING TEST
# =========================================================

@app.post("/embed-test")
async def embed_test(
    data: dict
):

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

        "text":
            text,

        "dimensions":
            len(embedding),

        "embedding":
            embedding

    }


# =========================================================
# VECTOR SEARCH
# =========================================================

@app.post("/search")
async def search_documents(
    data: dict
):

    query = data.get("query")


    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query is required."
        )


    # -----------------------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = generate_embedding(
        query,
        prefix="query"
    )


    if len(query_embedding) != 1024:

        raise HTTPException(
            status_code=500,
            detail=(
                "Expected 1024 dimensions, "
                f"got {len(query_embedding)}"
            )
        )


    # -----------------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------------

    try:

        response = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding":
                    query_embedding,

                "match_count":
                    10
            }
        ).execute()


        results = response.data or []


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )


    # -----------------------------------------------------
    # REMOVE DUPLICATE CHUNKS
    # -----------------------------------------------------

    unique_results = []

    seen = set()


    for result in results:

        key = (
            result.get("filename"),
            result.get("chunk_index")
        )


        if key in seen:

            continue


        seen.add(key)

        unique_results.append(result)


    # -----------------------------------------------------
    # RETURN TOP RESULTS
    # -----------------------------------------------------

    unique_results = unique_results[:5]


    return {

        "query":
            query,

        "results":
            unique_results

    }


# =========================================================
# RAG ASK
# =========================================================

@app.post("/ask")
async def ask_question(
    data: dict
):

    query = data.get("query")
    document_id = data.get("document_id")

    if not document_id:
       raise HTTPException(
        status_code=400,
        detail="document_id is required."
    )


    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query is required."
        )


    # -----------------------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = generate_embedding(
        query,
        prefix="query"
    )


    if len(query_embedding) != 1024:

        raise HTTPException(
            status_code=500,
            detail=(
                "Expected 1024 dimensions, "
                f"got {len(query_embedding)}"
            )
        )


    # -----------------------------------------------------
    # RETRIEVE DOCUMENT CHUNKS
    # -----------------------------------------------------

    try:

        response = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding":
                    query_embedding,

                "match_count":
                    10
            }
        ).execute()


        results = response.data or []


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )


    # -----------------------------------------------------
    # REMOVE DUPLICATE CHUNKS
    # -----------------------------------------------------

    unique_results = []

    seen = set()


    for result in results:

        filename = result.get(
            "filename"
        )

        chunk_index = result.get(
            "chunk_index"
        )

        key = (
            filename,
            chunk_index
        )


        if key in seen:

            continue


        seen.add(key)

        unique_results.append(result)


    # -----------------------------------------------------
    # SIMILARITY FILTER
    # -----------------------------------------------------

    filtered_results = []


    for result in unique_results:

        similarity = result.get(
            "similarity"
        )


        if similarity is None:

            filtered_results.append(
                result
            )

            continue


        # Keep reasonably relevant chunks
        if float(similarity) >= 0.70:

            filtered_results.append(
                result
            )


    # -----------------------------------------------------
    # USE TOP 5 RELEVANT CHUNKS
    # -----------------------------------------------------

    results = filtered_results[:5]


    # -----------------------------------------------------
    # NO RESULTS
    # -----------------------------------------------------

    if not results:

        return {

            "query":
                query,

            "answer":
                "I could not find that information "
                "in the uploaded document.",

            "sources":
                []

        }


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context_parts = []


    for result in results:

        filename = result.get(
            "filename",
            "Unknown document"
        )

        chunk_index = result.get(
            "chunk_index",
            "?"
        )

        content = result.get(
            "content",
            ""
        )


        if content:

            context_parts.append(

                f"""
DOCUMENT: {filename}
CHUNK: {chunk_index}

CONTENT:
{content}
"""

            )


    context = (
        "\n\n"
        "=============================="
        "\n\n"
    ).join(
        context_parts
    )


    # -----------------------------------------------------
    # DEBUG LOG
    # -----------------------------------------------------

    print(
        "========== RETRIEVED CHUNKS =========="
    )

    for result in results:

        print(
            "filename:",
            result.get("filename")
        )

        print(
            "chunk:",
            result.get("chunk_index")
        )

        print(
            "similarity:",
            result.get("similarity")
        )

        print(
            "--------------------------------------"
        )


    print(
        "========== CONTEXT =========="
    )

    print(
        context[:6000]
    )

    print(
        "========== END CONTEXT =========="
    )


    # =====================================================
    # GROQ PROMPT
    # =====================================================

    prompt = f"""
You are DocMind AI, an AI document knowledge assistant.

Your job is to answer the user's question using ONLY
the document context provided below.

IMPORTANT RULES:

1. Use only information from the document context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is present in the context, answer it directly.
5. If multiple chunks contain relevant information,
   combine them into one answer.
6. If the answer is not present in the context, say:
   "I could not find that information in the uploaded document."
7. Keep the answer clear and concise.
8. For lists, use bullet points.
9. Do not mention embeddings, vector databases,
   similarity scores, chunks, or internal RAG processing
   unless the user specifically asks about them.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

Give only the final answer.
"""


    # =====================================================
    # GROQ GENERATION
    # =====================================================

    try:

        completion = (
            groq_client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            "You are DocMind AI. "
                            "Answer using only "
                            "the supplied document context."
                    },

                    {
                        "role":
                            "user",

                        "content":
                            prompt
                    }

                ],

                temperature=0.2,

                max_completion_tokens=1000,

                reasoning_effort="low",

                include_reasoning=False

            )
        )


        # -------------------------------------------------
        # EXTRACT ANSWER
        # -------------------------------------------------

        message = (
            completion.choices[0].message
        )


        answer = (
            message.content or ""
        )


        # -------------------------------------------------
        # DEBUG GROQ
        # -------------------------------------------------

        print(
            "========== GROQ RESPONSE =========="
        )

        print(
            completion
        )

        print(
            "========== GROQ ANSWER =========="
        )

        print(
            repr(answer)
        )

        print(
            "========== END GROQ =========="
        )


        # -------------------------------------------------
        # EMPTY RESPONSE FALLBACK
        # -------------------------------------------------

        if not answer.strip():

            answer = (
                "I could not generate an answer "
                "from the retrieved document context."
            )


    except Exception as e:

        print(
            "========== GROQ ERROR =========="
        )

        print(
            str(e)
        )

        print(
            "========== END GROQ ERROR =========="
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Groq generation failed: "
                f"{str(e)}"
            )
        )


    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {

        "query":
            query,

        "answer":
            answer,

        "sources": [

            {

                "filename":
                    result.get("filename"),

                "chunk_index":
                    result.get("chunk_index"),

                "similarity":
                    result.get("similarity")

            }

            for result in results

        ]

    }

    


    

        
        
    
