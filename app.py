import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "The_Lightning_Thief.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

app = FastAPI(
    title="The Lightning Thief RAG",
    version="1.0.0"
)


# ============================================================
# API KEY
# ============================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY is not set."
    )


# ============================================================
# AI MODELS
# ============================================================

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.1,
    max_output_tokens=180
)

vector_store = None


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Book PDF not found at {PDF_PATH}"
        )

    print("Loading The Lightning Thief PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks:

        page = chunk.metadata.get("page")

        if page is not None:
            chunk.metadata["page_number"] = int(page) + 1

    print("Creating FAISS index...")

    store = FAISS.from_documents(
        chunks,
        embeddings
    )

    store.save_local(str(INDEX_DIR))

    print("FAISS index created successfully.")

    return store


# ============================================================
# LOAD EXISTING INDEX OR BUILD NEW INDEX
# ============================================================

def load_or_build_store():

    index_file = INDEX_DIR / "index.faiss"

    if INDEX_DIR.exists() and index_file.exists():

        print("Loading existing FAISS index...")

        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("FAISS index not found.")
    print("Building a new index...")

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    vector_store = load_or_build_store()

    print("======================================")
    print("Lightning Thief RAG is ready.")
    print("======================================")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    k: int = 2


class SourceItem(BaseModel):
    page: int | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ============================================================
# RAG PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You are a concise question-answering assistant for
Rick Riordan's The Lightning Thief.

Follow these rules strictly:

1. Answer ONLY from the provided book context.
2. Do not use outside knowledge.
3. Give a direct answer to the user's question.
4. Keep the answer to 1-3 sentences.
5. Do not repeat the question.
6. Do not explain your reasoning.
7. Do not mention "context", "retrieval", "RAG", or "chunks".
8. Do not add unnecessary background information.
9. If the question is unrelated to The Lightning Thief,
   answer exactly:
   "I don't know based on the provided book."
10. If the provided context does not contain enough information,
    answer exactly:
    "I don't know based on the provided book."

BOOK CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# CONVERT GEMINI RESPONSE TO STRING
# ============================================================

def extract_text(content) -> str:

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

            elif hasattr(item, "text"):

                text = getattr(item, "text")

                if text:
                    parts.append(str(text))

            else:

                parts.append(str(item))

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    index_file = BASE_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail="index.html is missing."
        )

    return FileResponse(index_file)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "book": PDF_PATH.name,
        "vector_index": INDEX_DIR.exists()
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    global vector_store

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Make sure FAISS is available
    if vector_store is None:

        vector_store = load_or_build_store()

    # Always use only 2 relevant chunks
    k = 2

    # ========================================================
    # RETRIEVAL
    # ========================================================

    try:

        docs = vector_store.similarity_search(
            question,
            k=k
        )

    except Exception as e:

        print(f"RETRIEVAL ERROR: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )

    # ========================================================
    # NO DOCUMENTS
    # ========================================================

    if not docs:

        return AskResponse(
            answer="I don't know based on the provided book.",
            sources=[]
        )

    # ========================================================
    # PREPARE CONTEXT
    # ========================================================

    context_parts = []
    sources = []

    for doc in docs:

        page = doc.metadata.get("page_number")

        page_label = (
            str(page)
            if page is not None
            else "unknown"
        )

        text = doc.page_content.strip()

        context_parts.append(
            f"[Book page {page_label}]\n{text}"
        )

        # Small source preview for frontend
        short_text = (
            text
            .replace("\n", " ")
            .strip()
        )

        if len(short_text) > 180:
            short_text = short_text[:180] + "..."

        sources.append(
            SourceItem(
                page=page,
                text=short_text
            )
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # ========================================================
    # CREATE PROMPT
    # ========================================================

    prompt = PROMPT.format_messages(
        context=context,
        question=question
    )

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        result = llm.invoke(prompt)

        answer = extract_text(
            result.content
        )

    except Exception as e:

        print(f"LLM ERROR: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    if not answer:

        answer = (
            "I don't know based on the provided book."
        )

    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return AskResponse(
        answer=answer,
        sources=sources
    )


# ============================================================
# REBUILD FAISS INDEX
# ============================================================

@app.post("/rebuild")
def rebuild_index():

    global vector_store

    try:

        vector_store = build_vector_store()

        return {
            "status": "rebuilt",
            "message": (
                "FAISS index rebuilt successfully."
            )
        }

    except Exception as e:

        print(f"REBUILD ERROR: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild index: {str(e)}"
        )
