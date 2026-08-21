🧠 DocMind AI — Backend

«An end-to-end RAG-powered document intelligence backend that allows users to upload PDF documents, retrieve relevant information using semantic vector search, and generate grounded answers with an LLM.»

🚀 Overview

DocMind AI is a Retrieval-Augmented Generation (RAG) backend built with FastAPI.

It processes uploaded PDF documents, converts their content into vector embeddings, stores them in Supabase with pgvector, retrieves the most relevant document chunks for a user's question, and generates answers using Groq.

The system also includes a custom RAG evaluation pipeline for measuring retrieval success, fact coverage, answer accuracy, and unsupported-query handling.

---

✨ Key Features

- 📄 PDF document upload and text extraction
- ✂️ Intelligent document chunking
- 🧠 Semantic embeddings using Hugging Face
- 🔎 Vector similarity search with pgvector
- 📚 Document-specific retrieval
- 🤖 Grounded AI answers using Groq
- 🔗 Retrieved source tracking
- 🧪 Custom RAG evaluation system
- 📊 Retrieval and fact-coverage metrics
- 🗂️ Document management
- 📈 Document analytics
- 🗑️ Document deletion
- 🌐 REST API built with FastAPI
- ☁️ Production deployment on Render

---

🧠 RAG Architecture

                 ┌──────────────────┐
                 │    PDF Upload    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ PDF Text Extract │
                 │    pdfplumber    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     Chunking     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Embeddings     │
                 │ Hugging Face E5  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Supabase pgvector│
                 │   Vector Store   │
                 └────────┬─────────┘
                          │
                    User Question
                          │
                          ▼
                 ┌──────────────────┐
                 │ Query Embedding  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Vector Similarity│
                 │     Search       │
                 └────────┬─────────┘
                          │
                    Top Relevant
                       Chunks
                          │
                          ▼
                 ┌──────────────────┐
                 │   Groq LLM       │
                 │ Context + Query  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Grounded Answer  │
                 │   + Sources      │
                 └──────────────────┘

---

🔄 How the RAG Pipeline Works

1. Document Ingestion

The "/upload" endpoint accepts a PDF and extracts readable text using "pdfplumber".

2. Chunking

Extracted text is divided into overlapping chunks so relevant information can be retrieved efficiently.

3. Embedding Generation

Each chunk is converted into a semantic vector using:

intfloat/multilingual-e5-large

4. Vector Storage

Embeddings and document metadata are stored in Supabase PostgreSQL using pgvector.

Each chunk is associated with its parent "document_id".

5. Query Retrieval

When a user asks a question, the question is converted into an embedding using the E5 query format.

The backend performs vector similarity search and retrieves the most relevant chunks belonging to the selected document.

6. Context Generation

The retrieved chunks are combined into a context that is passed to the LLM.

7. Grounded Generation

Groq generates the final response using only the retrieved document context.

If the information cannot be found, the system responds that the information is not available in the uploaded document.

---

🛠️ Tech Stack

Technology| Purpose
Python| Backend development
FastAPI| REST API
pdfplumber| PDF text extraction
Hugging Face| Text embeddings
"intfloat/multilingual-e5-large"| Embedding model
Supabase| PostgreSQL database
pgvector| Vector similarity search
Groq| LLM generation
Render| Backend deployment

---

🔌 API Endpoints

Health Check

GET /

Checks whether the backend is running.

Upload Document

POST /upload

Uploads and processes a PDF document.

Ask a Question

POST /ask

Generates a grounded answer from the selected document.

Example request:

{
  "document_id": "DOCUMENT_UUID",
  "query": "What skills are listed in the document?"
}

Vector Search

POST /search

Performs semantic document retrieval.

List Documents

GET /documents

Returns completed documents.

Delete Document

DELETE /documents/{document_id}

Deletes a document and its associated chunks.

Analytics

GET /analytics

Returns document and knowledge-base statistics.

RAG Evaluation

POST /evaluate

Runs custom evaluation tests against a selected document.

---

🧪 RAG Evaluation

DocMind AI includes a dedicated evaluation system instead of relying only on subjective chatbot testing.

Tests can define:

- User question
- Expected facts
- Expected behavior
- Retrieval success
- Retrieval similarity
- Fact coverage
- Overall test result

Example:

{
  "query": "What skills are listed in the document?",
  "expected_facts": [
    "Java",
    "OOP",
    "Collections",
    "SQL",
    "Spring Boot",
    "REST",
    "Docker",
    "Kafka"
  ],
  "expected_behavior": "answer"
}

The evaluation system can also test unsupported questions to verify that the model does not invent information.

Example Evaluation Result

Accuracy          100%
Retrieval         100%
Fact Coverage     100%

These results are based on the project's custom evaluation tests and should be treated as test-suite results rather than a universal accuracy guarantee.

---

🔐 Environment Variables

Create a ".env" file locally:

HF_TOKEN=your_huggingface_token
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key

Never commit real API keys or secrets to GitHub.

---

⚙️ Local Setup

1. Clone the repository

git clone https://github.com/santoshml-lab/DocMind-AI-Backend.git
cd DocMind-AI-Backend

2. Create a virtual environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Add your credentials to ".env".

5. Run the API

uvicorn main:app --reload

The API will be available locally at:

http://127.0.0.1:8000

---

🌐 Deployment

The backend is deployed using Render.

Live API:

"https://docmind-ai-backend-nwhv.onrender.com"

Frontend:

"https://doc-mind-ai-frontend.vercel.app"

---

📁 Project Structure

DocMind-AI-Backend/
│
├── main.py
├── requirements.txt
├── README.md
└── ...

---

🔒 RAG Safety & Grounding

The generation pipeline is designed to reduce hallucination by instructing the model to:

- Use only retrieved document context
- Avoid outside knowledge
- Avoid inventing facts
- Return a not-found response when the requested information is unavailable
- Answer only about the selected document

Document filtering is also applied during vector retrieval using "document_id".

---

🎯 Project Goal

The goal of DocMind AI is to demonstrate a complete production-style RAG workflow:

Document → Embedding → Vector Retrieval → Context → LLM → Grounded Answer → Evaluation

The project focuses not only on generating answers, but also on measuring whether the retrieval and generated answers are actually supported by the source document.

---

👨‍💻 Author

Santosh

AI/ML Developer 

GitHub:
https://github.com/santoshml-lab

---

⭐ Project

If you find this project interesting, consider giving the repository a ⭐.
