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
    title="Lightning Thief RAG",
    version="2.0.0"
)


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Add GOOGLE_API_KEY in Render Environment Variables."
    )


# ============================================================
# MODELS
# ============================================================

# Fast and reliable embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Flash model for faster answers
LLM_MODEL = "gemini-2.5-flash"


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)


llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.1,
    max_output_tokens=250,
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = None


def build_vector_store():
    """
    Build FAISS index from The Lightning Thief PDF.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"The PDF was not found:\n{PDF_PATH}\n\n"
            "Make sure The_Lightning_Thief.pdf is in the same "
            "folder as app.py."
        )

    print("Loading Lightning Thief PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} PDF pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Store page number internally.
    # It will NOT be sent to the user.
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


def load_or_build_store():
    """
    Load existing FAISS index.
    If it doesn't exist, build a new one.
    """

    index_file = INDEX_DIR / "index.faiss"

    if index_file.exists():

        print("Loading existing FAISS index...")

        try:
            store = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )

            print("FAISS index loaded.")

            return store

        except Exception as e:

            print(
                "Existing FAISS index could not be loaded."
            )

            print(str(e))

            print("Building a fresh index...")

            return build_vector_store()

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    print("----------------------------------------")
    print("Starting Lightning Thief RAG...")
    print("----------------------------------------")

    vector_store = load_or_build_store()

    print("----------------------------------------")
    print("Lightning Thief RAG is ready.")
    print("----------------------------------------")


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
    sources: List[SourceItem] = []


# ============================================================
# PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You are the question-answering assistant for
Rick Riordan's "The Lightning Thief".

Your ONLY job is to answer the user's question.

IMPORTANT RULES:

1. Use ONLY the provided book context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT invent information.
5. Answer ONLY what the user asked.
6. Keep the answer short and direct.
7. Normally answer in 1 to 3 sentences.
8. Do not repeat the question.
9. Do not mention "retrieved context".
10. Do not mention "chunks".
11. Do not mention "FAISS".
12. Do not mention pages.
13. Do not mention the RAG system.
14. Do not explain these instructions.
15. Do not output headings such as "Answer:".
16. Do not output bullet points unless the question specifically asks for a list.
17. Do not provide unrelated information.

If the answer cannot be found in the provided book context,
respond with exactly:

I don't know based on the book.

BOOK CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""
)


# ============================================================
# EXTRACT GEMINI TEXT
# ============================================================

def extract_text(content) -> str:
    """
    Gemini may return content as either:

        "normal string"

    or:

        [
            {"type": "text", "text": "..."}
        ]

    This function safely converts either format
    into a normal Python string.
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

        return " ".join(parts).strip()

    return str(content).strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer: str) -> str:
    """
    Remove accidental model formatting or instruction leakage.
    """

    if not answer:
        return "I don't know based on the book."

    answer = answer.strip()

    # Remove common unwanted prefixes
    prefixes = [
        "FINAL ANSWER:",
        "Final Answer:",
        "ANSWER:",
        "Answer:",
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # Remove accidental instruction leakage
    unwanted_phrases = [
        "No outside knowledge, guesses, invented",
        "No outside knowledge",
        "Use only the provided context",
        "Use ONLY the provided book context",
        "BOOK CONTEXT:",
        "USER QUESTION:",
    ]

    for phrase in unwanted_phrases:

        if phrase.lower() in answer.lower():

            # If the model returned prompt text instead
            # of an answer, don't show that to the user.
            return "I don't know based on the book."

    return answer.strip()


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
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    global vector_store

    question = request.question.strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # Make sure vector store exists
    # --------------------------------------------------------

    if vector_store is None:

        vector_store = load_or_build_store()

    # --------------------------------------------------------
    # Retrieve only a small number of relevant chunks
    # --------------------------------------------------------

    k = max(
        1,
        min(request.k, 3)
    )

    try:

        docs = vector_store.similarity_search(
            question,
            k=k
        )

    except Exception as e:

        print("Retrieval error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to search the book."
        )

    # --------------------------------------------------------
    # No relevant content
    # --------------------------------------------------------

    if not docs:

        return AskResponse(
            answer="I don't know based on the book.",
            sources=[]
        )

    # --------------------------------------------------------
    # Build clean context
    # --------------------------------------------------------

    context_parts = []

    for doc in docs:

        text = doc.page_content.strip()

        if text:

            context_parts.append(text)

    if not context_parts:

        return AskResponse(
            answer="I don't know based on the book.",
            sources=[]
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = PROMPT.format_messages(
        context=context,
        question=question
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        result = llm.invoke(prompt)

    except Exception as e:

        print("Gemini error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Gemini content can be a LIST.
    # Convert it to a normal string.
    # --------------------------------------------------------

    answer = extract_text(
        result.content
    )

    # --------------------------------------------------------
    # Clean answer
    # --------------------------------------------------------

    answer = clean_answer(
        answer
    )

    # --------------------------------------------------------
    # Return ONLY answer to frontend.
    #
    # sources are intentionally empty so the UI does not
    # display Book 1 / Book 2 / page information.
    # --------------------------------------------------------

    return AskResponse(
        answer=answer,
        sources=[]
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
            "message": "Lightning Thief FAISS index rebuilt successfully."
        }

    except Exception as e:

        print("Rebuild error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to rebuild the index."
        )
