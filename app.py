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
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "The_Lightning_Thief.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

app = FastAPI(
    title="Lightning Thief RAG",
    version="3.0.0"
)


# ============================================================
# API KEY
# ============================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set.")


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)

# Keep output small for faster response
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
    max_output_tokens=100
)

vector_store = None


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_vector_store():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Loading book...")

    loader = PyPDFLoader(
        str(PDF_PATH)
    )

    documents = loader.load()

    print(
        f"Loaded {len(documents)} pages."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=80,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    for chunk in chunks:

        page = chunk.metadata.get("page")

        if page is not None:
            chunk.metadata["page_number"] = (
                int(page) + 1
            )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating FAISS index...")

    store = FAISS.from_documents(
        chunks,
        embeddings
    )

    store.save_local(
        str(INDEX_DIR)
    )

    print("FAISS index ready.")

    return store


# ============================================================
# LOAD EXISTING INDEX
# ============================================================

def load_or_build_store():

    index_file = INDEX_DIR / "index.faiss"

    if (
        INDEX_DIR.exists()
        and index_file.exists()
    ):

        print("Loading existing FAISS index...")

        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    vector_store = load_or_build_store()

    print("================================")
    print("Lightning Thief RAG ONLINE")
    print("================================")


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
# STRICT ANSWER PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You answer questions ONLY about the book
"The Lightning Thief" by Rick Riordan.

BOOK CONTEXT:
{context}

QUESTION:
{question}

FOLLOW THESE RULES EXACTLY:

- Answer ONLY the question asked.
- Use ONLY the BOOK CONTEXT.
- Do NOT use outside knowledge.
- Do NOT guess.
- Do NOT add background information.
- Do NOT add extra facts.
- Do NOT explain your reasoning.
- Do NOT repeat the question.
- Do NOT summarize the context.
- Do NOT mention the context.
- Do NOT mention RAG, FAISS, AI, retrieval, chunks,
  embeddings, or sources.
- Do NOT say "according to the context".
- Give the shortest useful answer.
- Normally use ONE sentence.
- Use TWO sentences only when necessary.
- If the answer is not clearly supported by the
  BOOK CONTEXT, respond EXACTLY:

I don't know based on the provided book.

IMPORTANT:
If the question is about another book, movie, character,
person, website, or unrelated topic, respond EXACTLY:

I don't know based on the provided book.

FINAL ANSWER:
"""
)


# ============================================================
# EXTRACT GEMINI TEXT
# ============================================================

def extract_text(content):

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        result = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    result.append(
                        str(text)
                    )

            elif hasattr(item, "text"):

                text = getattr(
                    item,
                    "text"
                )

                if text:
                    result.append(
                        str(text)
                    )

        return " ".join(result).strip()

    return str(content).strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer):

    answer = answer.strip()

    unwanted_prefixes = [
        "FINAL ANSWER:",
        "Final Answer:",
        "Answer:",
        "ANSWER:"
    ]

    for prefix in unwanted_prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

    answer = answer.replace(
        "```",
        ""
    ).strip()

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

    return FileResponse(
        index_file
    )


# ============================================================
# HEALTH
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

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    if vector_store is None:

        vector_store = load_or_build_store()

    # --------------------------------------------------------
    # Retrieve ONLY 2 relevant chunks
    # --------------------------------------------------------

    try:

        docs = vector_store.similarity_search(
            question,
            k=2
        )

    except Exception as e:

        print(
            f"FAISS ERROR: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Book search failed."
        )

    # --------------------------------------------------------
    # No relevant content
    # --------------------------------------------------------

    if not docs:

        return AskResponse(
            answer=(
                "I don't know based on "
                "the provided book."
            ),
            sources=[]
        )

    # --------------------------------------------------------
    # Create SMALL context
    # --------------------------------------------------------

    context_parts = []
    sources = []

    for doc in docs:

        text = doc.page_content.strip()

        page = doc.metadata.get(
            "page_number"
        )

        context_parts.append(
            f"[Page {page}]\n{text}"
        )

        # Only page number goes to frontend
        sources.append(
            SourceItem(
                page=page,
                text=""
            )
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Ask Gemini
    # --------------------------------------------------------

    try:

        prompt = PROMPT.format_messages(
            context=context,
            question=question
        )

        result = llm.invoke(
            prompt
        )

        answer = extract_text(
            result.content
        )

        answer = clean_answer(
            answer
        )

    except Exception as e:

        print(
            f"GEMINI ERROR: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Answer generation failed."
        )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if not answer:

        answer = (
            "I don't know based on "
            "the provided book."
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return AskResponse(
        answer=answer,
        sources=sources
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

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
