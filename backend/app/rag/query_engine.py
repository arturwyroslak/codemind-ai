"""RAG query engine for code search and Q&A"""

from typing import List, Dict, Any, Optional
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """Engine for RAG-based code queries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize vector store (using Chroma for demo)
        self.vector_store = None
    
    async def query(
        self,
        query: str,
        repository_id: Optional[str],
        filters: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """Query codebase using RAG"""
        logger.info(f"Processing RAG query: {query}")
        
        # For demo purposes, return mock data
        # In production, implement actual vector search
        
        sources = [
            {
                "file": "app/main.py",
                "lines": "1-50",
                "content": "# Main application entry point...",
                "relevance": 0.95
            },
            {
                "file": "app/api/routes.py",
                "lines": "10-30",
                "content": "# API route definitions...",
                "relevance": 0.87
            }
        ]
        
        # Generate answer using LLM with retrieved context
        answer = await self._generate_answer(query, sources)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.91
        }
    
    async def _generate_answer(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """Generate answer from query and sources"""
        
        context = "\n\n".join([
            f"File: {s['file']}\n{s['content']}"
            for s in sources
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful code assistant.
            Answer questions based on the provided code context.
            Be specific and reference file names when relevant."""),
            ("user", """Context from codebase:
            {context}
            
            Question: {query}
            """)
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "context": context,
            "query": query
        })
        
        return response.content
    
    async def index_repository(
        self,
        repository_url: str,
        repository_id: str
    ) -> Dict[str, Any]:
        """Index a code repository"""
        logger.info(f"Indexing repository: {repository_url}")
        
        # Mock implementation for demo
        return {
            "files_indexed": 42,
            "chunks_created": 256
        }