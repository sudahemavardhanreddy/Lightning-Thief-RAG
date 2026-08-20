import os
from pathlib import Path

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
    version="4.0.0"
)


# ============================================================
# API KEY
# ============================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY is not set."
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
    temperature=0,
    max_output_tokens=160
)


vector_store = None


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Book not found: {PDF_PATH}"
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
        chunk_overlap=100,
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

    print(
        f"Created {len(chunks)} chunks."
    )

    store = FAISS.from_documents(
        chunks,
        embeddings
    )

    store.save_local(
        str(INDEX_DIR)
    )

    print("FAISS index created.")

    return store


# ============================================================
# LOAD OR BUILD INDEX
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

    print("Building FAISS index...")

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    global vector_store

    vector_store = load_or_build_store()

    print("===================================")
    print("LIGHTNING THIEF RAG ONLINE")
    print("===================================")


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    question: str


# ============================================================
# RESPONSE MODEL
# ============================================================

class AskResponse(BaseModel):

    answer: str


# ============================================================
# STRICT RAG PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You are a book question-answering assistant.

The only book you are allowed to answer questions about is:

"The Lightning Thief" by Rick Riordan.

You have been given relevant passages from the book.

BOOK PASSAGES:
{context}

USER QUESTION:
{question}

Follow these rules:

1. Answer the user's question directly.
2. Use ONLY the information contained in the book passages.
3. Never use outside knowledge.
4. Never guess.
5. Never invent facts.
6. Never discuss anything unrelated to the question.
7. Never repeat the book passages.
8. Never explain your reasoning.
9. Never mention RAG, AI, FAISS, embeddings, retrieval,
   passages, context, or sources.
10. Do not provide page numbers.
11. Do not provide a list of sources.
12. Do not say "according to the context".
13. Give a natural human answer.
14. Give enough information to actually answer the question.
15. Do NOT answer with only a person's name unless the
    question specifically asks for just a name.
16. Normally answer in 1-3 sentences.
17. If the question cannot be answered from the provided
    book passages, say exactly:

I don't know based on the provided book.

Examples:

Question:
Who is Percy Jackson?

Good answer:
Percy Jackson is the twelve-year-old protagonist of
The Lightning Thief and the son of Poseidon.

Question:
Who is Percy's father?

Good answer:
Percy's father is Poseidon.

Question:
Why is the master bolt important?

Good answer:
The master bolt is important because its theft threatens
to cause a war among the Greek gods.

Question:
Tell me about Harry Potter.

Good answer:
I don't know based on the provided book.

FINAL ANSWER:
"""
)


# ============================================================
# EXTRACT GEMINI RESPONSE
# ============================================================

def extract_text(content):

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(
                        str(text)
                    )

            elif hasattr(item, "text"):

                text = getattr(
                    item,
                    "text"
                )

                if text:
                    parts.append(
                        str(text)
                    )

            elif isinstance(item, str):

                parts.append(item)

        return " ".join(parts).strip()

    return str(content).strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer):

    answer = answer.strip()

    prefixes = [
        "FINAL ANSWER:",
        "Final Answer:",
        "ANSWER:",
        "Answer:"
    ]

    for prefix in prefixes:

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
# HOME
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
# ASK
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
    # Retrieve only 2 relevant chunks
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

    if not docs:

        return AskResponse(
            answer=(
                "I don't know based on "
                "the provided book."
            )
        )

    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content.strip()
        for doc in docs
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        prompt = PROMPT.format_messages(
            context=context,
            question=question
        )

        result = llm.invoke(prompt)

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
            detail="Failed to generate answer."
        )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not answer:

        answer = (
            "I don't know based on "
            "the provided book."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Return ONLY the answer.
    # No pages.
    # No sources.
    # No book1/book2.
    # --------------------------------------------------------

    return AskResponse(
        answer=answer
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
