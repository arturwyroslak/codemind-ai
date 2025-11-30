"""Code generation endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging

from app.agents.code_generator import CodeGenerator
from app.core.monitoring import llm_token_usage

logger = logging.getLogger(__name__)
router = APIRouter()


class CodeGenerationRequest(BaseModel):
    """Request model for code generation"""
    description: str = Field(..., description="Natural language description")
    language: str = Field(default="python", description="Target programming language")
    framework: Optional[str] = Field(None, description="Framework to use")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    style: Optional[str] = Field("clean", description="Code style preference")


class CodeGenerationResponse(BaseModel):
    """Response model for code generation"""
    status: str
    code: str
    language: str
    explanation: str
    suggestions: List[str]
    test_code: Optional[str] = None
    tokens_used: int


@router.post("/", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """
    Generate code from natural language description
    
    This endpoint uses advanced LLMs to convert natural language
    descriptions into working code with proper structure and documentation.
    """
    try:
        logger.info(f"Generating {request.language} code")
        logger.info(f"Description: {request.description[:100]}...")
        
        # Initialize code generator
        generator = CodeGenerator()
        
        # Generate code
        result = await generator.generate(
            description=request.description,
            language=request.language,
            framework=request.framework,
            context=request.context or {},
            style=request.style
        )
        
        # Record token usage
        llm_token_usage.labels(
            model=result.get("model", "unknown"),
            type="generation"
        ).inc(result.get("tokens_used", 0))
        
        response = CodeGenerationResponse(
            status="success",
            code=result["code"],
            language=request.language,
            explanation=result["explanation"],
            suggestions=result.get("suggestions", []),
            test_code=result.get("test_code"),
            tokens_used=result.get("tokens_used", 0)
        )
        
        logger.info("Code generation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}"
        )


@router.post("/refactor")
async def refactor_code(code: str, language: str, improvements: List[str]):
    """
    Refactor existing code based on suggested improvements
    """
    try:
        generator = CodeGenerator()
        
        result = await generator.refactor(
            code=code,
            language=language,
            improvements=improvements
        )
        
        return {
            "status": "success",
            "refactored_code": result["code"],
            "changes": result["changes"],
            "explanation": result["explanation"]
        }
        
    except Exception as e:
        logger.error(f"Code refactoring failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Code refactoring failed: {str(e)}"
        )


@router.post("/complete")
async def complete_code(code: str, language: str, cursor_position: int):
    """
    Provide code completion suggestions
    """
    try:
        generator = CodeGenerator()
        
        completions = await generator.complete(
            code=code,
            language=language,
            cursor_position=cursor_position
        )
        
        return {
            "status": "success",
            "completions": completions
        }
        
    except Exception as e:
        logger.error(f"Code completion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Code completion failed: {str(e)}"
        )