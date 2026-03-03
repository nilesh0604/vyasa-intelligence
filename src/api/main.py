"""FastAPI main application for Vyasa Intelligence."""

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Vyasa Intelligence API",
    description="A RAG system for querying the Mahabharata",
    version="0.1.0",
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
    # TODO: Implement RAG pipeline
    return QueryResponse(
        answer="This is a placeholder response. The RAG pipeline is not yet implemented.",
        sources=["Source 1", "Source 2"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec: B104
