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
    title="The Lightning Thief RAG",
    version="2.0.0"
)


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
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
    temperature=0.1,
    max_output_tokens=150
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = None


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_vector_store():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Book PDF not found: {PDF_PATH}"
        )

    print("Loading PDF...")

    loader = PyPDFLoader(
        str(PDF_PATH)
    )

    documents = loader.load()

    print(
        f"Loaded {len(documents)} pages."
    )

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

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    for chunk in chunks:

        page = chunk.metadata.get(
            "page"
        )

        if page is not None:
            chunk.metadata[
                "page_number"
            ] = int(page) + 1

    print("Creating FAISS index...")

    store = FAISS.from_documents(
        chunks,
        embeddings
    )

    store.save_local(
        str(INDEX_DIR)
    )

    print(
        "FAISS index created successfully."
    )

    return store


# ============================================================
# LOAD OR BUILD INDEX
# ============================================================

def load_or_build_store():

    index_file = (
        INDEX_DIR / "index.faiss"
    )

    if (
        INDEX_DIR.exists()
        and index_file.exists()
    ):

        print(
            "Loading existing FAISS index..."
        )

        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    print(
        "FAISS index not found."
    )

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    vector_store = load_or_build_store()

    print(
        "======================================"
    )

    print(
        "Lightning Thief RAG is ONLINE"
    )

    print(
        "======================================"
    )


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
# STRICT RAG PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You are the official question-answering assistant
for the book "The Lightning Thief" by Rick Riordan.

Your ONLY source of truth is the BOOK CONTEXT below.

STRICT RULES:

1. Answer the user's exact question.
2. Use ONLY information explicitly supported by the
   provided BOOK CONTEXT.
3. Do NOT use your general knowledge.
4. Do NOT guess.
5. Do NOT invent information.
6. Do NOT repeat the retrieved context.
7. Do NOT mention the retrieval process.
8. Do NOT mention RAG, chunks, embeddings, FAISS,
   context, or sources.
9. Keep the answer short and direct.
10. Normally answer in 1 to 3 sentences.
11. If the question asks for a simple fact,
    answer with just that fact.
12. If the question cannot be answered from the
    BOOK CONTEXT, respond EXACTLY with:

I don't know based on the provided book.

IMPORTANT:

A question being related to a famous character or topic
does NOT mean you should answer it from general knowledge.

For example:

Question:
"Tell me about Harry Potter."

Correct response:
"I don't know based on the provided book."

Question:
"What is Percy Jackson's father?"

If the BOOK CONTEXT says Percy is the son of Poseidon,
answer:
"Percy's father is Poseidon."

Do not add unrelated information.

--------------------------------

BOOK CONTEXT:

{context}

--------------------------------

USER QUESTION:

{question}

--------------------------------

FINAL ANSWER:
"""
)


# ============================================================
# EXTRACT TEXT FROM GEMINI RESPONSE
# ============================================================

def extract_text(content) -> str:

    if content is None:
        return ""

    # Normal Gemini string response
    if isinstance(content, str):
        return content.strip()

    # Sometimes Gemini/LangChain returns a list
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

        return " ".join(parts).strip()

    return str(content).strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer: str) -> str:

    answer = answer.strip()

    # Remove accidental labels
    prefixes = [
        "FINAL ANSWER:",
        "Answer:",
        "ANSWER:"
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

    # Remove markdown code fences
    answer = answer.replace(
        "```",
        ""
    ).strip()

    # If model somehow returned an empty answer
    if not answer:

        return (
            "I don't know based on the provided book."
        )

    return answer


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    index_file = (
        BASE_DIR / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail="index.html is missing."
        )

    return FileResponse(
        index_file
    )


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
        )
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

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = (
        request.question
        .strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # Make sure vector store exists
    # --------------------------------------------------------

    if vector_store is None:

        vector_store = (
            load_or_build_store()
        )

    # --------------------------------------------------------
    # Retrieve only the best 2 chunks
    # --------------------------------------------------------

    try:

        docs = vector_store.similarity_search(
            question,
            k=2
        )

    except Exception as e:

        print(
            f"Retrieval error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to search the book."
        )

    # --------------------------------------------------------
    # Nothing retrieved
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
    # Prepare ONLY retrieved context
    # --------------------------------------------------------

    context_parts = []

    sources = []

    for doc in docs:

        page = doc.metadata.get(
            "page_number"
        )

        text = (
            doc.page_content
            .strip()
        )

        # Context sent to Gemini
        context_parts.append(
            f"[Page {page}]\n{text}"
        )

        # Short source information
        preview = (
            text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 150:

            preview = (
                preview[:150]
                + "..."
            )

        sources.append(
            SourceItem(
                page=page,
                text=preview
            )
        )

    context = (
        "\n\n---\n\n"
        .join(context_parts)
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = PROMPT.format_messages(
        context=context,
        question=question
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

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
            f"Gemini error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate "
                "the answer."
            )
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
    # Return ONLY answer + small source info
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

        vector_store = (
            build_vector_store()
        )

        return {
            "status": "rebuilt",
            "message": (
                "FAISS index rebuilt successfully "
                "from The Lightning Thief PDF."
            )
        }

    except Exception as e:

        print(
            f"Rebuild error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to rebuild index: {str(e)}"
            )
        )
