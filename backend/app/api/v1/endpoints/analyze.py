"""Code analysis endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import time

from app.agents.orchestrator import AgentOrchestrator
from app.core.monitoring import agent_execution_time

logger = logging.getLogger(__name__)
router = APIRouter()


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis"""
    code: str = Field(..., description="Source code to analyze")
    language: str = Field(..., description="Programming language")
    agents: List[str] = Field(
        default=["security", "performance", "architecture"],
        description="Agents to use for analysis"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for analysis"
    )


class AgentResult(BaseModel):
    """Individual agent result"""
    agent: str
    status: str
    findings: List[Dict[str, Any]]
    summary: str
    execution_time: float


class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis"""
    status: str
    results: List[AgentResult]
    total_execution_time: float
    overall_score: float
    recommendations: List[str]


@router.post("/", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """
    Analyze code using multi-agent system
    
    This endpoint orchestrates multiple specialized agents to analyze code:
    - Security Agent: Identifies vulnerabilities
    - Performance Agent: Analyzes efficiency
    - Architecture Agent: Reviews design patterns
    - Documentation Agent: Checks documentation quality
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting code analysis for {request.language} code")
        logger.info(f"Using agents: {request.agents}")
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        
        # Run analysis
        results = await orchestrator.analyze_code(
            code=request.code,
            language=request.language,
            agents=request.agents,
            context=request.context or {}
        )
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Record metrics for each agent
        for result in results:
            agent_execution_time.labels(
                agent_type=result["agent"]
            ).observe(result["execution_time"])
        
        # Calculate overall score (0-100)
        overall_score = orchestrator.calculate_overall_score(results)
        
        # Generate recommendations
        recommendations = orchestrator.generate_recommendations(results)
        
        response = CodeAnalysisResponse(
            status="success",
            results=[AgentResult(**r) for r in results],
            total_execution_time=total_time,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        logger.info(f"Analysis completed in {total_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/quick")
async def quick_analyze(request: CodeAnalysisRequest):
    """
    Quick code analysis with single agent
    
    Faster analysis using only one agent for quick feedback
    """
    try:
        # Use only security agent for quick analysis
        request.agents = ["security"]
        
        orchestrator = AgentOrchestrator()
        results = await orchestrator.analyze_code(
            code=request.code,
            language=request.language,
            agents=request.agents,
            context=request.context or {}
        )
        
        return {
            "status": "success",
            "result": results[0] if results else None
        }
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Quick analysis failed: {str(e)}"
        )