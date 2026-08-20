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
    title="Lightning Thief Assistant",
    version="3.0.0",
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

# Fast embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"

# IMPORTANT:
# Do NOT use gemini-2.5-flash.
LLM_MODEL = "gemini-3.6-flash"


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)


# Keep generation short for faster responses
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=180,
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = None


def build_vector_store():
    """
    Load the PDF, split it into chunks,
    create embeddings and save the FAISS index.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"The book PDF was not found:\n{PDF_PATH}"
        )

    print("Loading The Lightning Thief PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} PDF pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Add readable page numbers to metadata
    for chunk in chunks:
        page = chunk.metadata.get("page")

        if page is not None:
            chunk.metadata["page_number"] = int(page) + 1

    print("Creating embeddings...")

    store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    store.save_local(str(INDEX_DIR))

    print("FAISS index created successfully.")

    return store


def load_or_build_store():
    """
    Load existing FAISS index.
    If it doesn't exist, create it.
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

    print("FAISS index not found.")

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    try:
        vector_store = load_or_build_store()
        print("Lightning Thief RAG is ready.")

    except Exception as e:
        print(f"Startup error: {repr(e)}")
        raise


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
# STRICT ANSWER PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You answer questions ONLY about the book
"The Lightning Thief" by Rick Riordan.

Your task is very simple:

Answer ONLY the user's question.

STRICT RULES:

1. Use ONLY the supplied book text.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT invent information.
5. Ignore unrelated retrieved text.
6. Do NOT mention retrieved text.
7. Do NOT mention context.
8. Do NOT mention chunks.
9. Do NOT mention embeddings.
10. Do NOT mention FAISS.
11. Do NOT mention RAG.
12. Do NOT mention sources.
13. Do NOT mention page numbers.
14. Do NOT repeat the question.
15. Do NOT discuss unrelated characters or events.
16. Do NOT provide a general summary unless specifically asked.
17. Answer in 1-3 short sentences.
18. Be direct and specific.
19. If the question is unrelated to The Lightning Thief, say:
"I don't know based on the book."
20. If the supplied text does not contain enough information to answer,
say exactly:
"I don't know based on the book."

IMPORTANT:
The retrieved text can contain information unrelated to the question.
Use ONLY the part that directly answers the question.

BOOK TEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(content: Any) -> str:
    """
    Safely extract text from Gemini's response.
    """

    if content is None:
        return ""

    # Normal string
    if isinstance(content, str):
        return content.strip()

    # List response
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

    return str(content).strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer: str) -> str:

    if not answer:
        return "I don't know based on the book."

    answer = answer.strip()

    # Remove accidental prefixes
    prefixes = [
        "Answer:",
        "Answer -",
        "Answer:",
        "Response:",
        "Response -",
    ]

    for prefix in prefixes:

        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Remove markdown fences if Gemini adds them
    answer = answer.replace("```text", "")
    answer = answer.replace("```", "")
    answer = answer.strip()

    # Internal words that should NEVER appear
    forbidden_phrases = [
        "retrieved context",
        "book page",
        "retrieved text",
        "context:",
        "chunk",
        "chunks",
        "faiss",
        "embedding",
        "embeddings",
        "rag",
        "vector store",
        "vectorstore",
        "source:",
        "sources:",
    ]

    lower_answer = answer.lower()

    for phrase in forbidden_phrases:

        if phrase in lower_answer:
            return "I don't know based on the book."

    # Remove accidental question repetition
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

    # Make sure vector store exists
    if vector_store is None:
        vector_store = load_or_build_store()

    # --------------------------------------------------------
    # Keep retrieval small for speed
    # --------------------------------------------------------

    k = max(
        1,
        min(request.k, 2)
    )

    try:

        # ----------------------------------------------------
        # Retrieve relevant documents
        # ----------------------------------------------------

        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=k
        )

        # ----------------------------------------------------
        # Remove weak matches
        # ----------------------------------------------------

        relevant_docs = []

        for doc, score in results:

            # Higher score = more relevant
            if score >= 0.30:
                relevant_docs.append(doc)

        # If relevance filtering removed everything,
        # use the best result only if it is reasonably close.
        if not relevant_docs and results:

            best_doc, best_score = results[0]

            if best_score >= 0.20:
                relevant_docs = [best_doc]

        # ----------------------------------------------------
        # No relevant information
        # ----------------------------------------------------

        if not relevant_docs:

            return AskResponse(
                answer="I don't know based on the book.",
                sources=[]
            )

        # ----------------------------------------------------
        # Build SMALL context
        # ----------------------------------------------------

        context_parts = []

        for doc in relevant_docs:

            text = doc.page_content.strip()

            if text:
                context_parts.append(text)

        context = "\n\n".join(context_parts)

        if not context:

            return AskResponse(
                answer="I don't know based on the book.",
                sources=[]
            )

        # ----------------------------------------------------
        # Limit context size
        # ----------------------------------------------------

        # This helps response speed and keeps Gemini focused.
        context = context[:7000]

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        prompt = PROMPT.format_messages(
            context=context,
            question=question,
        )

        result = llm.invoke(prompt)

        # ----------------------------------------------------
        # Extract Gemini response
        # ----------------------------------------------------

        answer = extract_text(result.content)

        # ----------------------------------------------------
        # Clean response
        # ----------------------------------------------------

        answer = clean_answer(answer)

        # ----------------------------------------------------
        # Sources
        #
        # They remain available to the frontend API,
        # but they are NOT included in the answer.
        # ----------------------------------------------------

        sources = []

        for doc in relevant_docs:

            page = doc.metadata.get("page_number")

            text = (
                doc.page_content
                .strip()
                .replace("\n", " ")
            )

            sources.append(
                SourceItem(
                    page=page,
                    text=text[:250],
                )
            )

        # ----------------------------------------------------
        # Return ONLY the answer + hidden source data
        # ----------------------------------------------------

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
