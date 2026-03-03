"""FastAPI main application for Vyasa Intelligence."""

# Disable tracing BEFORE any other imports
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from typing import List, Optional  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# Load environment variables
load_dotenv()

# Ensure tracing is still disabled after loading .env
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.pipeline import RAGPipeline  # noqa: E402

app = FastAPI(
    title="Vyasa Intelligence API",
    description="A RAG system for querying the Mahabharata",
    version="0.1.0",
)

# Initialize RAG pipeline
pipeline = RAGPipeline(
    chroma_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
    bm25_path=os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl"),
    llm_provider=os.getenv("LLM_PROVIDER", "groq"),
    llm_model=os.getenv("LLM_MODEL"),
    enable_cache=True,
    cache_type="memory",
    enable_guardrails=True,
    enable_tracing=False,
)


class QueryRequest(BaseModel):
    question: str
    user_role: Optional[str] = "public"
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Vyasa Intelligence API is running"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the Mahabharata knowledge base."""
    # Process query through RAG pipeline
    result = pipeline.query(
        question=request.question,
        user_role=request.user_role,
        top_k=request.top_k,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        retrieval_time_ms=result.get("retrieval_time", 0) * 1000,
        generation_time_ms=result.get("generation_time", 0) * 1000,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec: B104
