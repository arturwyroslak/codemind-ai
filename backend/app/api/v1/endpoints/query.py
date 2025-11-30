"""RAG query endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.rag.query_engine import RAGQueryEngine

logger = logging.getLogger(__name__)
router = APIRouter()


class RAGQueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="Natural language query")
    repository_id: Optional[str] = Field(None, description="Repository to search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    top_k: int = Field(5, description="Number of results to return")


class RAGQueryResponse(BaseModel):
    """Response model for RAG query"""
    status: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


@router.post("/", response_model=RAGQueryResponse)
async def query_codebase(request: RAGQueryRequest):
    """
    Query codebase using RAG (Retrieval-Augmented Generation)
    
    This endpoint allows natural language queries over indexed codebases,
    providing context-aware answers with source references.
    """
    try:
        logger.info(f"Processing RAG query: {request.query[:100]}...")
        
        # Initialize RAG engine
        rag_engine = RAGQueryEngine()
        
        # Execute query
        result = await rag_engine.query(
            query=request.query,
            repository_id=request.repository_id,
            filters=request.filters or {},
            top_k=request.top_k
        )
        
        response = RAGQueryResponse(
            status="success",
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
        logger.info("RAG query completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@router.post("/index")
async def index_repository(repository_url: str, repository_id: str):
    """
    Index a code repository for RAG queries
    """
    try:
        logger.info(f"Indexing repository: {repository_url}")
        
        rag_engine = RAGQueryEngine()
        
        result = await rag_engine.index_repository(
            repository_url=repository_url,
            repository_id=repository_id
        )
        
        return {
            "status": "success",
            "repository_id": repository_id,
            "files_indexed": result["files_indexed"],
            "chunks_created": result["chunks_created"]
        }
        
    except Exception as e:
        logger.error(f"Repository indexing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Repository indexing failed: {str(e)}"
        )