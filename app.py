import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Lightning_Thief_KT(1).pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

app = FastAPI(title="The Lightning Thief RAG", version="1.0.0")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your environment before starting the app.")

# ---------------------------
# RAG configuration
# ---------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.2,
)

vector_store = None


def build_vector_store():
    """Load the book, split it into chunks, create embeddings and build FAISS."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge PDF not found at {PDF_PATH}. Put Lightning_Thief_KT(1).pdf beside app.py."
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Keep page numbers in metadata so the UI can show useful sources.
    for chunk in chunks:
        page = chunk.metadata.get("page")
        if page is not None:
            chunk.metadata["page_number"] = int(page) + 1

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    return store


def load_or_build_store():
    """Load an existing FAISS index, otherwise build one from the PDF."""
    if INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return build_vector_store()


@app.on_event("startup")
def startup_event():
    global vector_store
    vector_store = load_or_build_store()


class AskRequest(BaseModel):
    question: str
    k: int = 4


class SourceItem(BaseModel):
    page: int | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful RAG assistant for Rick Riordan's The Lightning Thief.

Use ONLY the retrieved context below to answer the user's question.
Do not invent facts that are not supported by the context.
If the answer cannot be found in the context, say:
"I don't know based on the provided book."

Give a clear, concise answer. If useful, mention the relevant chapter/page information
available in the source metadata.

Retrieved context:
{context}

Question:
{question}
"""
)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "book": PDF_PATH.name,
        "vector_index": INDEX_DIR.exists(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    global vector_store

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vector_store is None:
        vector_store = load_or_build_store()

    k = max(1, min(request.k, 8))
    docs = vector_store.similarity_search(question, k=k)

    if not docs:
        return AskResponse(
            answer="I don't know based on the provided book.",
            sources=[],
        )

    context_parts = []
    sources = []

    for doc in docs:
        page = doc.metadata.get("page_number")
        context_parts.append(
            f"[Book page {page if page is not None else 'unknown'}]\n{doc.page_content}"
        )
        sources.append(
            SourceItem(
                page=page,
                text=doc.page_content[:500].replace("\n", " "),
            )
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT.format_messages(context=context, question=question)
    result = llm.invoke(prompt)

    return AskResponse(
        answer=result.content,
        sources=sources,
    )


@app.post("/rebuild")
def rebuild_index():
    """Rebuild the vector database after changing the source PDF."""
    global vector_store
    vector_store = build_vector_store()
    return {"status": "rebuilt", "message": "FAISS index rebuilt from the PDF."}
