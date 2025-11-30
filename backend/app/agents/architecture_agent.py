"""Architecture analysis agent"""

from typing import List, Dict, Any
import logging
import ast

from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class ArchitectureAgent:
    """Agent specialized in architecture and design pattern analysis"""
    
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
        """Analyze code architecture and design patterns"""
        logger.info("Running architecture analysis")
        
        findings = []
        
        # Check SOLID principles
        findings.extend(self._check_solid_principles(code, language))
        
        # Check for code smells
        findings.extend(self._detect_code_smells(code, language))
        
        return findings
    
    def _check_solid_principles(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Check SOLID principles compliance"""
        findings = []
        
        if language.lower() == "python":
            try:
                tree = ast.parse(code)
                
                # Single Responsibility Principle - check class size
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                        if method_count > 10:
                            findings.append({
                                "severity": "medium",
                                "category": "srp_violation",
                                "line": node.lineno,
                                "description": f"Class '{node.name}' has {method_count} methods - possible SRP violation",
                                "recommendation": "Consider splitting into smaller, focused classes"
                            })
            
            except SyntaxError:
                pass
        
        return findings
    
    def _detect_code_smells(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect common code smells"""
        findings = []
        
        # Long methods
        lines = code.split('\n')
        if len(lines) > 100:
            findings.append({
                "severity": "low",
                "category": "long_method",
                "description": f"Method/function is {len(lines)} lines long",
                "recommendation": "Consider breaking into smaller functions"
            })
        
        return findings
    
    def generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Generate architecture analysis summary"""
        if not findings:
            return "Code follows good architectural practices."
        
        return f"Found {len(findings)} architectural improvements.."