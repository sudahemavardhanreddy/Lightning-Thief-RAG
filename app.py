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
    version="4.0.0"
)


# ============================================================
# GOOGLE API KEY
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

# Embedding model used for FAISS retrieval
EMBEDDING_MODEL = "models/gemini-embedding-001"

# IMPORTANT:
# Do NOT use gemini-2.5-flash here.
#
# This is the current stable Flash model we use.
LLM_MODEL = "gemini-3.6-flash"


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    max_output_tokens=300,
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
            f"The book PDF was not found:\n{PDF_PATH}\n\n"
            "Make sure The_Lightning_Thief.pdf is in the same "
            "folder as app.py."
        )

    print("Loading The Lightning Thief PDF...")

    loader = PyPDFLoader(
        str(PDF_PATH)
    )

    documents = loader.load()

    print(
        f"Loaded {len(documents)} PDF pages."
    )

    # Smaller chunks improve relevance
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
            ""
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # Store page number internally.
    # It will NEVER be shown to the user.

    for chunk in chunks:

        page = chunk.metadata.get("page")

        if page is not None:

            chunk.metadata["page_number"] = (
                int(page) + 1
            )

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
# LOAD OR BUILD VECTOR STORE
# ============================================================

def load_or_build_store():

    index_file = INDEX_DIR / "index.faiss"
    pkl_file = INDEX_DIR / "index.pkl"

    if (
        index_file.exists()
        and pkl_file.exists()
    ):

        print(
            "Loading existing FAISS index..."
        )

        try:

            store = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )

            print(
                "FAISS index loaded successfully."
            )

            return store

        except Exception as e:

            print(
                "Could not load existing FAISS index."
            )

            print(
                repr(e)
            )

            print(
                "Building a new FAISS index..."
            )

    return build_vector_store()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global vector_store

    print("----------------------------------------")
    print("Starting Lightning Thief Assistant...")
    print("----------------------------------------")

    vector_store = load_or_build_store()

    print("----------------------------------------")
    print("Lightning Thief Assistant is READY.")
    print("----------------------------------------")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):

    question: str

    # Keep this small for faster retrieval.
    k: int = 2


class SourceItem(BaseModel):

    page: int | None = None
    text: str


class AskResponse(BaseModel):

    answer: str

    # Kept for API compatibility.
    # Frontend does NOT display these.
    sources: List[SourceItem] = []


# ============================================================
# PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You answer questions about Rick Riordan's
"The Lightning Thief".

Your task is extremely simple:

Answer ONLY the user's question.

RULES:

- Use ONLY the supplied book context.
- Do not use outside knowledge.
- Do not guess.
- Do not invent information.
- Ignore context that is unrelated to the question.
- Give the most relevant answer only.
- Keep the answer concise.
- Usually answer in 1 to 3 complete sentences.
- Make sure every sentence is COMPLETE.
- Never stop in the middle of a word.
- Never stop in the middle of a sentence.
- Do not repeat the question.
- Do not mention the context.
- Do not mention retrieval.
- Do not mention chunks.
- Do not mention FAISS.
- Do not mention embeddings.
- Do not mention RAG.
- Do not mention page numbers.
- Do not provide sources.
- Do not provide unrelated information.
- Do not add headings such as "Answer:".

If the context does NOT contain enough information
to answer the question, respond EXACTLY:

I don't know based on the book.

BOOK CONTEXT:
{context}

USER QUESTION:
{question}

Return ONLY the final answer.
"""
)


# ============================================================
# EXTRACT GEMINI TEXT
# ============================================================

def extract_text(content: Any) -> str:

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
                    parts.append(
                        str(text)
                    )

            else:

                text = getattr(
                    item,
                    "text",
                    None
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

    if not answer:
        return "I don't know based on the book."

    answer = answer.strip()

    # Remove common prefixes
    prefixes = [
        "FINAL ANSWER:",
        "Final Answer:",
        "ANSWER:",
        "Answer:",
        "Response:",
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

    # Remove accidental model leakage
    forbidden_phrases = [
        "BOOK CONTEXT:",
        "USER QUESTION:",
        "retrieved context",
        "retrieved book content",
        "FAISS",
        "embedding",
        "chunks",
        "RAG system",
        "No outside knowledge",
    ]

    lower_answer = answer.lower()

    for phrase in forbidden_phrases:

        if phrase.lower() in lower_answer:

            return (
                "I don't know based on the book."
            )

    # Remove accidental surrounding quotes
    if (
        len(answer) >= 2
        and answer[0] == '"'
        and answer[-1] == '"'
    ):
        answer = answer[1:-1].strip()

    return answer


# ============================================================
# CHECK IF ANSWER LOOKS INCOMPLETE
# ============================================================

def answer_looks_incomplete(answer: str) -> bool:

    if not answer:
        return True

    answer = answer.strip()

    # Obvious unfinished word
    if answer.endswith("-"):
        return True

    # Obvious unfinished punctuation
    if answer.endswith(
        (
            ",",
            ":",
            ";",
            "(",
            "/",
        )
    ):
        return True

    # Very short fragment
    if len(answer) < 8:
        return True

    return False


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    context: str,
    question: str
) -> str:

    prompt = PROMPT.format_messages(
        context=context,
        question=question,
    )

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

        # ----------------------------------------------------
        # If Gemini accidentally stops mid-word/sentence,
        # make ONE short retry.
        # ----------------------------------------------------

        if answer_looks_incomplete(
            answer
        ):

            retry_prompt = ChatPromptTemplate.from_template(
                """
Answer the user's question using ONLY the
book context below.

Give ONE complete, concise answer.
Do not stop mid-word.
Do not stop mid-sentence.
Do not add unrelated information.
Do not mention the context.

BOOK CONTEXT:
{context}

QUESTION:
{question}

Return only the complete answer.
"""
            ).format_messages(
                context=context,
                question=question,
            )

            retry_result = llm.invoke(
                retry_prompt
            )

            retry_answer = extract_text(
                retry_result.content
            )

            retry_answer = clean_answer(
                retry_answer
            )

            if retry_answer:
                answer = retry_answer

        return answer

    except Exception as e:

        print(
            "Gemini generation error:"
        )

        print(
            repr(e)
        )

        raise


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
            and (
                INDEX_DIR / "index.faiss"
            ).exists()
        ),
        "model": LLM_MODEL,
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

    question = (
        request.question.strip()
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # MAKE SURE VECTOR STORE EXISTS
    # --------------------------------------------------------

    if vector_store is None:

        vector_store = (
            load_or_build_store()
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    # Only retrieve 2 chunks.
    # This makes the response faster and
    # prevents unrelated book content.

    k = max(
        1,
        min(
            request.k,
            2
        )
    )

    try:

        docs = (
            vector_store
            .similarity_search(
                question,
                k=k
            )
        )

    except Exception as e:

        print(
            "FAISS retrieval error:"
        )

        print(
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to search the book."
        )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not docs:

        return AskResponse(
            answer=(
                "I don't know based on the book."
            ),
            sources=[]
        )

    # --------------------------------------------------------
    # BUILD FOCUSED CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for doc in docs:

        text = (
            doc.page_content
            .strip()
        )

        if text:

            context_parts.append(
                text
            )

    if not context_parts:

        return AskResponse(
            answer=(
                "I don't know based on the book."
            ),
            sources=[]
        )

    # Keep context compact
    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            context,
            question
        )

    except Exception as e:

        print(
            "Gemini/RAG error:"
        )

        print(
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer."
        )

    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    answer = clean_answer(
        answer
    )

    if not answer:

        answer = (
            "I don't know based on the book."
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We intentionally return EMPTY sources.
    #
    # This prevents the frontend from showing:
    #
    # Book 1
    # Book 2
    # Page 1
    # Page 2
    # chunks
    # etc.
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

        vector_store = (
            build_vector_store()
        )

        return {
            "status": "rebuilt",
            "message": (
                "FAISS index rebuilt successfully."
            )
        }

    except Exception as e:

        print(
            "Rebuild error:"
        )

        print(
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to rebuild the index."
        )
