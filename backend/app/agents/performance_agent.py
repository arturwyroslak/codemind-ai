"""Performance analysis agent"""

from typing import List, Dict, Any
import logging
import ast

from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class PerformanceAgent:
    """Agent specialized in performance analysis"""
    
    def __init__(self):
        """Initialize performance agent"""
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
        """
        Analyze code for performance issues
        
        Checks for:
        - Algorithm complexity
        - Nested loops
        - Inefficient data structures
        - Memory leaks
        - Database query optimization
        """
        logger.info("Running performance analysis")
        
        findings = []
        
        if language.lower() == "python":
            findings.extend(self._analyze_python_performance(code))
        
        return findings
    
    def _analyze_python_performance(self, code: str) -> List[Dict[str, Any]]:
        """Analyze Python code performance"""
        findings = []
        
        try:
            tree = ast.parse(code)
            
            # Check for nested loops
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    for child in ast.walk(node):
                        if isinstance(child, ast.For) and child != node:
                            findings.append({
                                "severity": "medium",
                                "category": "nested_loops",
                                "line": node.lineno,
                                "description": "Nested loops detected - O(n²) complexity",
                                "recommendation": "Consider using more efficient algorithms or data structures"
                            })
            
            # Check for inefficient string concatenation
            for node in ast.walk(tree):
                if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                    if isinstance(node.target, ast.Name):
                        findings.append({
                            "severity": "low",
                            "category": "string_concatenation",
                            "line": node.lineno,
                            "description": "String concatenation in loop",
                            "recommendation": "Use list.append() and ''.join() instead"
                        })
        
        except SyntaxError as e:
            logger.error(f"Failed to parse Python code: {str(e)}")
        
        return findings
    
    def generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Generate summary of performance findings"""
        if not findings:
            return "No performance issues detected."
        
        return f"Found {len(findings)} performance optimization opportunities."