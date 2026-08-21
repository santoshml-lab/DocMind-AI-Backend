from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import io
import os
import uuid
import re
import unicodedata


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

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    # Normalize all whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary punctuation
    text = re.sub(r"[^\w\s.+#/-]", "", text)

    return text.strip()


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

        # Remove extra dimension if returned as [[...]]
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
# CREATE DOCUMENT RECORD
# =========================================================

def create_document_record(
    document_id,
    filename,
    pages
):

    try:

        response = supabase.table(
            "documents"
        ).insert({

            "id": document_id,

            "filename": filename,

            "status": "processing",

            "pages": pages,

            "chunks_count": 0

        }).execute()

        return response.data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to create document record: "
                f"{str(e)}"
            )
        )


# =========================================================
# UPDATE DOCUMENT STATUS
# =========================================================

def update_document_status(
    document_id,
    status,
    chunks_count=None
):

    try:

        update_data = {
            "status": status
        }

        if chunks_count is not None:

            update_data["chunks_count"] = chunks_count

        supabase.table(
            "documents"
        ).update(
            update_data
        ).eq(
            "id",
            document_id
        ).execute()

    except Exception as e:

        print(
            "========== DOCUMENT STATUS UPDATE ERROR =========="
        )

        print(
            str(e)
        )

        print(
            "==================================================="
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


    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )


    # -----------------------------------------------------
    # CREATE DOCUMENT ID
    # -----------------------------------------------------

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
    # CREATE DOCUMENT RECORD
    # -----------------------------------------------------

    create_document_record(
        document_id=document_id,
        filename=file.filename,
        pages=page_count
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
                    f"Expected 1024 dimensions, "
                    f"got {len(embedding)}"
                )


            supabase.table(
                "document_chunks"
            ).insert({

                "document_id":
                    document_id,

                "filename":
                    file.filename,

                "chunk_index":
                    index,

                "content":
                    chunk,

                "embedding":
                    embedding

            }).execute()


            stored_chunks += 1


        # -------------------------------------------------
        # MARK DOCUMENT AS COMPLETED
        # -------------------------------------------------

        update_document_status(
            document_id=document_id,
            status="completed",
            chunks_count=stored_chunks
        )


    except Exception as e:

        # -------------------------------------------------
        # MARK DOCUMENT AS FAILED
        # -------------------------------------------------

        update_document_status(
            document_id=document_id,
            status="failed",
            chunks_count=stored_chunks
        )


        # -------------------------------------------------
        # CLEANUP PARTIAL CHUNKS
        # -------------------------------------------------

        try:

            supabase.table(
                "document_chunks"
            ).delete().eq(
                "document_id",
                document_id
            ).execute()

        except Exception as cleanup_error:

            print(
                "========== CHUNK CLEANUP ERROR =========="
            )

            print(
                str(cleanup_error)
            )

            print(
                "=========================================="
            )


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
            stored_chunks,

        "stored_chunks":
            stored_chunks,

        "status":
            "completed"

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

    document_id = data.get(
        "document_id"
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
    # VECTOR SEARCH
    # -----------------------------------------------------

    try:

        rpc_params = {

            "query_embedding":
                query_embedding,

            "match_count":
                10,

            "filter_document_id":
                document_id

        }


        response = supabase.rpc(
            "match_document_chunks",
            rpc_params
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

        unique_results.append(
            result
        )


    # -----------------------------------------------------
    # RETURN TOP RESULTS
    # -----------------------------------------------------

    unique_results = (
        unique_results[:5]
    )


    return {

        "query":
            query,

        "document_id":
            document_id,

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

    document_id = data.get(
        "document_id"
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

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
    # CHECK DOCUMENT EXISTS
    # -----------------------------------------------------

    try:

        document_response = (
            supabase
            .table("documents")
            .select(
                "id, filename, status, pages, chunks_count"
            )
            .eq(
                "id",
                document_id
            )
            .limit(1)
            .execute()
        )


        document_data = (
            document_response.data or []
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document lookup failed: "
                f"{str(e)}"
            )
        )


    if not document_data:

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )


    document = document_data[0]


    # -----------------------------------------------------
    # CHECK DOCUMENT STATUS
    # -----------------------------------------------------

    if document.get("status") != "completed":

        raise HTTPException(
            status_code=409,
            detail=(
                "Document is not ready yet. "
                f"Current status: "
                f"{document.get('status')}"
            )
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
                    10,

                "filter_document_id":
                    document_id

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

        unique_results.append(
            result
        )


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


        try:

            similarity_value = float(
                similarity
            )

        except (TypeError, ValueError):

            continue


        if similarity_value >= 0.70:

            filtered_results.append(
                result
            )


    # -----------------------------------------------------
    # TOP 5 RELEVANT CHUNKS
    # -----------------------------------------------------

    results = (
        filtered_results[:5]
    )


    # -----------------------------------------------------
    # NO RELEVANT RESULTS
    # -----------------------------------------------------

    if not results:

        return {

            "query":
                query,

            "document_id":
                document_id,

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
            document.get(
                "filename",
                "Unknown document"
            )
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


    # =====================================================
    # DEBUG LOG
    # =====================================================

    print(
        "========== DOCUMENT =========="
    )

    print(
        "document_id:",
        document_id
    )

    print(
        "filename:",
        document.get("filename")
    )

    print(
        "status:",
        document.get("status")
    )

    print(
        "pages:",
        document.get("pages")
    )

    print(
        "chunks:",
        document.get("chunks_count")
    )


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
4. If the answer is present in the context,
   answer it directly.
5. If multiple document chunks contain relevant
   information, combine them into one answer.
6. If the answer is not present in the context, say:

"I could not find that information in the uploaded document."

7. Keep the answer clear and concise.
8. For lists, use bullet points.
9. Do not mention embeddings, vector databases,
   similarity scores, chunks, or internal RAG processing
   unless the user specifically asks about them.
10. Answer only about the selected document.

SELECTED DOCUMENT:

Filename:
{document.get("filename")}

Document ID:
{document_id}

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
            groq_client
            .chat
            .completions
            .create(

                model=GROQ_MODEL,

                messages=[

                    {

                        "role":
                            "system",

                        "content":
                            (
                                "You are DocMind AI. "
                                "Answer using only "
                                "the supplied document "
                                "context."
                            )

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
            completion
            .choices[0]
            .message
        )


        answer = (
            message.content or ""
        )


        # -------------------------------------------------
        # DEBUG GROQ
        # -------------------------------------------------

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

        "document_id":
            document_id,

        "document":
            {

                "filename":
                    document.get("filename"),

                "status":
                    document.get("status"),

                "pages":
                    document.get("pages"),

                "chunks_count":
                    document.get("chunks_count")

            },

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

# =========================================================
# LIST COMPLETED DOCUMENTS
# =========================================================

@app.get("/documents")
async def get_documents():

    try:

        response = (
            supabase
            .table("documents")
            .select(
                "id, filename, status, pages, chunks_count"
            )
            .eq(
                "status",
                "completed"
            )
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        return {
            "documents": response.data or []
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch documents: {str(e)}"
        )





            
            

# =========================================================
# DELETE DOCUMENT
# =========================================================

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):

    # -----------------------------------------------------
    # CHECK DOCUMENT EXISTS
    # -----------------------------------------------------

    try:

        document_response = (
            supabase
            .table("documents")
            .select("id, filename")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )

        document_data = document_response.data or []

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document lookup failed: {str(e)}"
        )

    if not document_data:

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    # -----------------------------------------------------
    # DELETE CHUNKS
    # -----------------------------------------------------

    try:

        supabase.table(
            "document_chunks"
        ).delete().eq(
            "document_id",
            document_id
        ).execute()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document chunks: {str(e)}"
        )

    # -----------------------------------------------------
    # DELETE DOCUMENT RECORD
    # -----------------------------------------------------

    try:

        supabase.table(
            "documents"
        ).delete().eq(
            "id",
            document_id
        ).execute()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "message": "Document deleted successfully 🗑️",
        "document_id": document_id,
        "filename": document_data[0]["filename"]
    }

# =========================================================
# DOCUMENT ANALYTICS
# =========================================================

@app.get("/analytics")
async def get_analytics():

    try:

        # -------------------------------------------------
        # DOCUMENT STATS
        # -------------------------------------------------

        documents_response = (
            supabase
            .table("documents")
            .select(
                "id, status, pages, chunks_count"
            )
            .execute()
        )

        documents = (
            documents_response.data or []
        )


        total_documents = len(documents)

        completed_documents = sum(
            1
            for doc in documents
            if doc.get("status") == "completed"
        )

        processing_documents = sum(
            1
            for doc in documents
            if doc.get("status") == "processing"
        )

        failed_documents = sum(
            1
            for doc in documents
            if doc.get("status") == "failed"
        )


        total_pages = sum(
            int(doc.get("pages") or 0)
            for doc in documents
        )

        total_chunks = sum(
            int(doc.get("chunks_count") or 0)
            for doc in documents
        )


        # -------------------------------------------------
        # KNOWLEDGE BASE HEALTH
        # -------------------------------------------------

        if failed_documents > 0:

            health = "attention_needed"

        elif processing_documents > 0:

            health = "processing"

        else:

            health = "healthy"


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "documents": {

                "total":
                    total_documents,

                "completed":
                    completed_documents,

                "processing":
                    processing_documents,

                "failed":
                    failed_documents

            },

            "knowledge_base": {

                "total_pages":
                    total_pages,

                "total_chunks":
                    total_chunks,

                "health":
                    health

            }

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Failed to generate analytics: "
                f"{str(e)}"
            )

        )

# =========================================================
# RAG EVALUATION V3
# =========================================================

@app.post("/evaluate")
async def evaluate_rag(data: dict):

    document_id = data.get("document_id")
    tests = data.get("tests", [])

    # =====================================================
    # VALIDATION
    # =====================================================

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="document_id is required."
        )

    if not tests:
        raise HTTPException(
            status_code=400,
            detail="tests are required."
        )

    results = []

    passed = 0

    total_fact_checks = 0
    matched_fact_checks = 0

    retrieval_success_count = 0

    not_found_tests = 0
    correct_not_found_count = 0

    # NEW
    similarity_values = []


    # =====================================================
    # RUN TESTS
    # =====================================================

    for index, test in enumerate(tests):

        query = test.get("query")

        expected_facts = test.get(
            "expected_facts",
            []
        )

        expected_behavior = test.get(
            "expected_behavior",
            "answer"
        )

        if not query:
            continue


        try:

            # =================================================
            # QUERY EMBEDDING
            # =================================================

            query_embedding = generate_embedding(
                query,
                prefix="query"
            )


            if len(query_embedding) != 1024:

                raise ValueError(
                    f"Expected 1024 dimensions, "
                    f"got {len(query_embedding)}"
                )


            # =================================================
            # VECTOR SEARCH
            # =================================================

            response = supabase.rpc(
                "match_document_chunks",
                {
                    "query_embedding":
                        query_embedding,

                    "match_count":
                        10,

                    "filter_document_id":
                        document_id
                }
            ).execute()


            retrieved = response.data or []


            # =================================================
            # REMOVE DUPLICATES
            # =================================================

            unique_results = []

            seen = set()


            for result in retrieved:

                key = (
                    result.get("filename"),
                    result.get("chunk_index")
                )


                if key in seen:
                    continue


                seen.add(key)

                unique_results.append(
                    result
                )


            # =================================================
            # SIMILARITY FILTER
            # =================================================

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


                try:

                    similarity_value = float(
                        similarity
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    continue


                if similarity_value >= 0.70:

                    filtered_results.append(
                        result
                    )


            # =================================================
            # TOP 5 RESULTS
            # =================================================

            results_for_context = (
                filtered_results[:5]
            )


            # =================================================
            # RETRIEVAL METRICS
            # =================================================

            retrieval_success = (
                len(results_for_context) > 0
            )


            if retrieval_success:

                retrieval_success_count += 1


            # =================================================
            # TOP SIMILARITY
            # =================================================

            similarities = []


            for result in results_for_context:

                similarity = result.get(
                    "similarity"
                )


                if similarity is not None:

                    try:

                        similarities.append(
                            float(similarity)
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        pass


            top_similarity = (
                max(similarities)
                if similarities
                else None
            )


            # =================================================
            # AVERAGE SIMILARITY DATA
            # =================================================

            if top_similarity is not None:

                similarity_values.append(
                    top_similarity
                )


            # =================================================
            # BUILD CONTEXT
            # =================================================

            context_parts = []


            for result in results_for_context:

                content = result.get(
                    "content",
                    ""
                )


                if content:

                    context_parts.append(
                        content
                    )


            context = "\n\n".join(
                context_parts
            )


            # =================================================
            # GENERATE ANSWER
            # =================================================

            if not results_for_context:

                answer = (
                    "I could not find that information "
                    "in the uploaded document."
                )


            else:

                prompt = f"""
You are DocMind AI.

Answer the user's question using ONLY
the provided document context.

IMPORTANT RULES:

1. Never use outside knowledge.
2. Never invent information.
3. If the information is not present,
   say exactly:

"I could not find that information in the uploaded document."

4. Keep the answer clear and concise.
5. Answer only from the document context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

Give only the final answer.
"""


                completion = (
                    groq_client
                    .chat
                    .completions
                    .create(

                        model=GROQ_MODEL,

                        messages=[

                            {
                                "role":
                                    "system",

                                "content":
                                    (
                                        "You are DocMind AI. "
                                        "Answer only from "
                                        "the supplied "
                                        "document context."
                                    )
                            },

                            {
                                "role":
                                    "user",

                                "content":
                                    prompt
                            }

                        ],

                        temperature=0.0,

                        max_completion_tokens=500,

                        reasoning_effort="low",

                        include_reasoning=False

                    )
                )


                answer = (
                    completion
                    .choices[0]
                    .message
                    .content or ""
                )


                if not answer.strip():

                    answer = (
                        "I could not generate an answer "
                        "from the retrieved document context."
                    )


            # =================================================
            # FACT COVERAGE
            # =================================================

            answer_normalized = normalize_text(answer)

             matched_facts = []

              for fact in expected_facts:

              total_fact_checks += 1

              fact_normalized = normalize_text(fact)

           if fact_normalized in answer_normalized:

              matched_facts.append(fact)

              matched_fact_checks += 1  


                

                    


            


            if expected_facts:

                fact_coverage = (
                    len(matched_facts)
                    / len(expected_facts)
                    * 100
                )

            else:

                fact_coverage = None


            # =================================================
            # UNSUPPORTED QUERY
            # =================================================

            if expected_behavior == "not_found":

                not_found_tests += 1


                is_correct_not_found = (
                    "could not find that information"
                    in answer_lower
                )


                if is_correct_not_found:

                    correct_not_found_count += 1


                passed_test = (
                    is_correct_not_found
                )


            # =================================================
            # NORMAL ANSWER
            # =================================================

            else:

                if expected_facts:

                    passed_test = (
                        fact_coverage == 100
                    )

                else:

                    passed_test = (
                        bool(answer.strip())
                    )


            # =================================================
            # PASS COUNT
            # =================================================

            if passed_test:

                passed += 1


            # =================================================
            # STORE TEST RESULT
            # =================================================

            results.append({

                "test_number":
                    index + 1,

                "query":
                    query,

                "answer":
                    answer,

                "expected_facts":
                    expected_facts,

                "matched_facts":
                    matched_facts,

                "fact_coverage":
                    (
                        round(
                            fact_coverage,
                            2
                        )
                        if fact_coverage is not None
                        else None
                    ),

                "expected_behavior":
                    expected_behavior,

                "retrieval_success":
                    retrieval_success,

                "top_similarity":
                    top_similarity,

                "passed":
                    passed_test

            })


        # =====================================================
        # TEST ERROR
        # =====================================================

        except Exception as e:

            results.append({

                "test_number":
                    index + 1,

                "query":
                    query,

                "answer":
                    "",

                "expected_facts":
                    expected_facts,

                "matched_facts":
                    [],

                "fact_coverage":
                    0,

                "expected_behavior":
                    expected_behavior,

                "retrieval_success":
                    False,

                "top_similarity":
                    None,

                "passed":
                    False,

                "error":
                    str(e)

            })


    # =========================================================
    # FINAL METRICS
    # =========================================================

    total_tests = len(results)


    accuracy = (
        passed
        / total_tests
        * 100
        if total_tests
        else 0
    )


    retrieval_success_rate = (
        retrieval_success_count
        / total_tests
        * 100
        if total_tests
        else 0
    )


    answer_fact_coverage = (
        matched_fact_checks
        / total_fact_checks
        * 100
        if total_fact_checks
        else 0
    )


    unsupported_query_accuracy = (
        correct_not_found_count
        / not_found_tests
        * 100
        if not_found_tests
        else None
    )


    # =========================================================
    # AVERAGE RETRIEVAL SIMILARITY
    # =========================================================

    average_similarity = (

        sum(similarity_values)
        / len(similarity_values)

        if similarity_values
        else 0

    )


    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    return {

        "document_id":
            document_id,

        "total_tests":
            total_tests,

        "passed":
            passed,

        "failed":
            total_tests - passed,

        "accuracy":
            round(
                accuracy,
                2
            ),

        "retrieval_success_rate":
            round(
                retrieval_success_rate,
                2
            ),

        "answer_fact_coverage":
            round(
                answer_fact_coverage,
                2
            ),

        "unsupported_query_accuracy":
            (
                round(
                    unsupported_query_accuracy,
                    2
                )
                if unsupported_query_accuracy
                is not None
                else None
            ),

        "average_similarity":
            round(
                average_similarity * 100,
                2
            ),

        "results":
            results

    }













            

        
            

    


    

        
        
    
