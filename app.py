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
# ENVIRONMENT
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
        "GOOGLE_API_KEY is not set. "
        "Add it to your environment before starting the app."
    )


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)


llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.2,
)


vector_store = None


# ============================================================
# BUILD FAISS VECTOR STORE
# ============================================================

def build_vector_store():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Book PDF not found at {PDF_PATH}. "
            "Put The_Lightning_Thief.pdf beside app.py."
        )

    print("Loading PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} PDF pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
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

    # Add human-readable page numbers
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

    print("FAISS index created successfully.")

    return store


# ============================================================
# LOAD EXISTING INDEX OR BUILD NEW ONE
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

    print("FAISS index not found. Building a new one...")

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    vector_store = load_or_build_store()

    print("RAG system is ready.")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    k: int = 4


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
You are a helpful RAG assistant for Rick Riordan's
The Lightning Thief.

Use ONLY the retrieved context below to answer the user's question.

Do not invent facts that are not supported by the context.

If the answer cannot be found in the context, say:

"I don't know based on the provided book."

Give a clear, concise answer.

Retrieved context:
{context}

Question:
{question}
"""
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_text(content) -> str:
    """
    Convert Gemini/LangChain response content into
    a normal Python string.

    Gemini can sometimes return content as:
        "some text"

    or as:
        [
            {"type": "text", "text": "some text"}
        ]

    This function handles both formats.
    """

    # Normal string response
    if isinstance(content, str):
        return content.strip()

    # List of content blocks
    if isinstance(content, list):

        parts = []

        for item in content:

            # Dictionary content block
            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

            # Other content object
            else:

                # Some LangChain content objects may
                # contain a text attribute
                if hasattr(item, "text"):

                    text = getattr(item, "text")

                    if text:
                        parts.append(str(text))

                else:
                    parts.append(str(item))

        return "\n".join(parts).strip()

    # Fallback
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
            detail="index.html is missing. Put index.html beside app.py.",
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
            detail="Question cannot be empty.",
        )

    # Make sure vector store exists
    if vector_store is None:

        vector_store = load_or_build_store()

    # Limit k between 1 and 8
    k = max(
        1,
        min(request.k, 8),
    )

    try:

        # ====================================================
        # RETRIEVE RELEVANT DOCUMENTS
        # ====================================================

        docs = vector_store.similarity_search(
            question,
            k=k,
        )

        if not docs:

            return AskResponse(
                answer="I don't know based on the provided book.",
                sources=[],
            )

        # ====================================================
        # PREPARE CONTEXT
        # ====================================================

        context_parts = []
        sources = []

        for doc in docs:

            page = doc.metadata.get("page_number")

            page_label = (
                str(page)
                if page is not None
                else "unknown"
            )

            context_parts.append(
                f"[Book page {page_label}]\n"
                f"{doc.page_content}"
            )

            sources.append(
                SourceItem(
                    page=page,
                    text=doc.page_content[:500].replace(
                        "\n",
                        " ",
                    ),
                )
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        # ====================================================
        # CREATE PROMPT
        # ====================================================

        prompt = PROMPT.format_messages(
            context=context,
            question=question,
        )

        # ====================================================
        # CALL GEMINI
        # ====================================================

        result = llm.invoke(prompt)

        # ====================================================
        # IMPORTANT FIX
        # ====================================================
        #
        # Gemini may return:
        #
        # result.content = "text"
        #
        # OR:
        #
        # result.content = [
        #     {"type": "text", "text": "..."}
        # ]
        #
        # Convert either format into a string.
        # ====================================================

        answer = extract_text(
            result.content
        )

        if not answer:

            answer = (
                "I don't know based on the provided book."
            )

        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return AskResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as e:

        print(
            f"ERROR while processing question: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}",
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
                "FAISS index rebuilt from "
                "The_Lightning_Thief.pdf."
            ),
        }

    except Exception as e:

        print(
            f"ERROR while rebuilding index: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild index: {str(e)}",
        )
