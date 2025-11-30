"""Code generation agent"""

from typing import Dict, Any, List, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Agent for generating and refactoring code"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3
        )
    
    async def generate(
        self,
        description: str,
        language: str,
        framework: Optional[str],
        context: Dict[str, Any],
        style: str
    ) -> Dict[str, Any]:
        """Generate code from description"""
        logger.info(f"Generating {language} code")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert {language} developer.
            Generate clean, efficient, well-documented code.
            Follow {style} coding style and best practices.
            {f'Use {framework} framework.' if framework else ''}
            
            Include:
            1. Complete, working code
            2. Proper error handling
            3. Clear comments and docstrings
            4. Type hints (if applicable)
            """),
            ("user", "Generate code for: {description}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"description": description})
        
        code = response.content
        
        return {
            "code": code,
            "explanation": f"Generated {language} code based on description",
            "suggestions": [
                "Review and test the generated code",
                "Add additional error handling as needed",
                "Customize to your specific requirements"
            ],
            "test_code": None,
            "tokens_used": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            "model": settings.OPENAI_MODEL
        }
    
    async def refactor(
        self,
        code: str,
        language: str,
        improvements: List[str]
    ) -> Dict[str, Any]:
        """Refactor existing code"""
        logger.info("Refactoring code")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert code refactoring assistant."),
            ("user", """Refactor this {language} code with these improvements:
            {improvements}
            
            Original code:
            {code}
            """)
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "code": code,
            "language": language,
            "improvements": "\n".join(improvements)
        })
        
        return {
            "code": response.content,
            "changes": improvements,
            "explanation": "Code refactored with requested improvements"
        }
    
    async def complete(
        self,
        code: str,
        language: str,
        cursor_position: int
    ) -> List[str]:
        """Provide code completions"""
        logger.info("Generating code completions")
        
        # Simplified completion logic
        return [
            "Completion suggestion 1",
            "Completion suggestion 2",
            "Completion suggestion 3"
        ]