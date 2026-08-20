import os
from pathlib import Path
from typing import List, Any

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
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "The_Lightning_Thief.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

app = FastAPI(
    title="The Lightning Thief RAG",
    version="2.0.0",
)


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Add GOOGLE_API_KEY to your Render environment variables."
    )


# ============================================================
# MODELS
# ============================================================

# Embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"

# IMPORTANT:
# Do NOT use gemini-2.5-flash.
# Render's current API response says to use gemini-3.6-flash.
LLM_MODEL = "gemini-3.6-flash"


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)


llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=250,
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = None


def build_vector_store():
    """
    Read the PDF, split it into chunks, create embeddings,
    and save the FAISS index.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Book PDF not found at: {PDF_PATH}"
        )

    print("Loading The Lightning Thief PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} PDF pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
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

    # Add readable page numbers
    for chunk in chunks:
        page = chunk.metadata.get("page")

        if page is not None:
            chunk.metadata["page_number"] = int(page) + 1

    print("Creating FAISS embeddings...")

    store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    store.save_local(str(INDEX_DIR))

    print("FAISS index created successfully.")

    return store


def load_or_build_store():
    """
    Load existing FAISS index if available.
    Otherwise build it from the PDF.
    """

    index_file = INDEX_DIR / "index.faiss"
    pkl_file = INDEX_DIR / "index.pkl"

    if index_file.exists() and pkl_file.exists():

        print("Loading existing FAISS index...")

        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("FAISS index not found. Building index...")

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    try:
        vector_store = load_or_build_store()
        print("RAG system is ready.")

    except Exception as e:
        print(f"Startup error: {e}")
        raise


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    k: int = 3


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
You are a question-answering assistant for the book
"The Lightning Thief" by Rick Riordan.

Your ONLY job is to answer the user's question using the
retrieved book content.

STRICT RULES:

1. Answer ONLY the user's question.
2. Use ONLY information supported by the retrieved context.
3. Do NOT use outside knowledge.
4. Do NOT mention the retrieved context.
5. Do NOT mention chunks.
6. Do NOT mention embeddings.
7. Do NOT mention FAISS.
8. Do NOT mention RAG.
9. Do NOT mention page numbers.
10. Do NOT list sources.
11. Do NOT summarize unrelated parts of the book.
12. Do NOT repeat the question.
13. Keep the answer short and direct.
14. Normally answer in 1-3 sentences.
15. If the retrieved context does not contain enough information
    to answer the question, respond exactly:

"I don't know based on the book."

IMPORTANT:
The retrieved context may contain unrelated information.
Ignore anything that does not directly help answer the question.

Retrieved book content:
{context}

User question:
{question}

Answer:
"""
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(content: Any) -> str:
    """
    Gemini can return content either as a normal string
    or as a list of content blocks.

    This function safely converts either format into a string.
    """

    if content is None:
        return ""

    # Normal string
    if isinstance(content, str):
        return content.strip()

    # Gemini content blocks
    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

            else:

                text = getattr(item, "text", None)

                if text:
                    parts.append(str(text))

        return " ".join(parts).strip()

    # Fallback
    return str(content).strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer: str) -> str:

    answer = answer.strip()

    # Remove accidental prefixes
    prefixes = [
        "Answer:",
        "Answer -",
        "Answer:",
        "Response:",
    ]

    for prefix in prefixes:

        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Prevent the model from dumping internal information
    forbidden_phrases = [
        "retrieved context:",
        "book page",
        "chunk",
        "faiss",
        "embedding",
        "rag",
    ]

    lower_answer = answer.lower()

    for phrase in forbidden_phrases:

        if phrase in lower_answer:
            return "I don't know based on the book."

    if not answer:
        return "I don't know based on the book."

    return answer


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
        "vector_index": (
            INDEX_DIR.exists()
            and (INDEX_DIR / "index.faiss").exists()
        ),
        "model": LLM_MODEL,
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

    # Load index if necessary
    if vector_store is None:

        vector_store = load_or_build_store()

    # Keep retrieval small for faster response
    k = max(
        1,
        min(request.k, 3)
    )

    try:

        # ----------------------------------------------------
        # RETRIEVE ONLY A FEW RELEVANT CHUNKS
        # ----------------------------------------------------

        docs = vector_store.similarity_search(
            question,
            k=k,
        )

        if not docs:

            return AskResponse(
                answer="I don't know based on the book.",
                sources=[]
            )

        # ----------------------------------------------------
        # BUILD SMALL CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for doc in docs:

            content = doc.page_content.strip()

            if content:

                context_parts.append(content)

        context = "\n\n---\n\n".join(context_parts)

        if not context:

            return AskResponse(
                answer="I don't know based on the book.",
                sources=[]
            )

        # ----------------------------------------------------
        # ASK GEMINI
        # ----------------------------------------------------

        prompt = PROMPT.format_messages(
            context=context,
            question=question,
        )

        result = llm.invoke(prompt)

        # ----------------------------------------------------
        # FIX GEMINI LIST RESPONSE
        # ----------------------------------------------------

        answer = extract_text(
            result.content
        )

        answer = clean_answer(answer)

        # ----------------------------------------------------
        # SOURCES
        #
        # Keep these in API response for the frontend if needed,
        # but DO NOT include them inside the answer.
        # ----------------------------------------------------

        sources = []

        for doc in docs:

            page = doc.metadata.get(
                "page_number"
            )

            text = (
                doc.page_content
                .strip()
                .replace("\n", " ")
            )

            sources.append(
                SourceItem(
                    page=page,
                    text=text[:300],
                )
            )

        return AskResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as e:

        print(
            f"Gemini/RAG error: {repr(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer."
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
            "message": "FAISS index rebuilt successfully."
        }

    except Exception as e:

        print(
            f"Rebuild error: {repr(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to rebuild FAISS index."
        )
