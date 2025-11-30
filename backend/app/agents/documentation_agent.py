"""Documentation analysis agent"""

from typing import List, Dict, Any
import logging
import ast
import re

from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentationAgent:
    """Agent specialized in documentation quality analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1
        )
    
    async def analyze(
        self,
        code: str,
        language: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze code documentation quality"""
        logger.info("Running documentation analysis")
        
        findings = []
        
        if language.lower() == "python":
            findings.extend(self._analyze_python_docs(code))
        
        return findings
    
    def _analyze_python_docs(self, code: str) -> List[Dict[str, Any]]:
        """Analyze Python documentation"""
        findings = []
        
        try:
            tree = ast.parse(code)
            
            # Check for missing docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        findings.append({
                            "severity": "low",
                            "category": "missing_docstring",
                            "line": node.lineno,
                            "description": f"Missing docstring for {node.name}",
                            "recommendation": "Add descriptive docstring"
                        })
        
        except SyntaxError:
            pass
        
        return findings
    
    def generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Generate documentation summary"""
        if not findings:
            return "Code is well-documented."
        
        return f"Found {len(findings)} documentation improvements."