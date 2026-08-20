```python
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
    version="1.0.0",
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


# Lower output tokens = faster + shorter answers
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.1,
    max_output_tokens=180,
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
            "",
        ],
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
        embeddings,
    )

    store.save_local(str(INDEX_DIR))

    print("FAISS index created.")

    return store


# ============================================================
# LOAD OR BUILD INDEX
# ============================================================

def load_or_build_store():

    index_file = INDEX_DIR / "index.faiss"

    if INDEX_DIR.exists() and index_file.exists():

        print("Loading existing FAISS index...")

        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    vector_store = load_or_build_store()

    print("Lightning Thief RAG is ready.")


# ============================================================
# DATA MODELS
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

IMPORTANT RULES:

1. Answer ONLY using the retrieved book context.
2. Do NOT use outside knowledge.
3. If the question is unrelated to The Lightning Thief,
   say exactly:
   "I don't know based on the provided book."
4. If the retrieved context does not contain enough
   information, say:
   "I don't know based on the provided book."
5. Never guess or invent information.
6. Keep the answer SHORT and DIRECT.
7. Usually answer in 1-3 sentences.
8. Do not repeat the question.
9. Do not explain your reasoning.
10. Do not mention the retrieved context.

BOOK CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# EXTRACT GEMINI TEXT
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
        "vector_index": INDEX_DIR.exists(),
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    global vector_store

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if vector_store is None:

        vector_store = load_or_build_store()

    # --------------------------------------------------------
    # Retrieve only a small number of relevant chunks
    # --------------------------------------------------------

    k = 2

    try:

        docs = vector_store.similarity_search(
            question,
            k=k,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )

    # --------------------------------------------------------
    # No useful documents
    # --------------------------------------------------------

    if not docs:

        return AskResponse(
            answer="I don't know based on the provided book.",
            sources=[]
        )

    # --------------------------------------------------------
    # Build small context
    # --------------------------------------------------------

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

        # Only show a SHORT source preview
        short_text = (
            text.replace("\n", " ")
            .strip()
        )

        if len(short_text) > 180:
            short_text = short_text[:180] + "..."

        sources.append(
            SourceItem(
                page=page,
                text=short_text,
            )
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    prompt = PROMPT.format_messages(
        context=context,
        question=question,
    )

    try:

        result = llm.invoke(prompt)

        answer = extract_text(
            result.content
        )

    except Exception as e:

        print(
            f"LLM ERROR: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not answer:

        answer = (
            "I don't know based on the provided book."
        )

    # --------------------------------------------------------
    # Return clean response
    # --------------------------------------------------------

    return AskResponse(
        answer=answer,
        sources=sources,
    )


# ============================================================
# REBUILD INDEX
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
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild index: {str(e)}"
        )
```
