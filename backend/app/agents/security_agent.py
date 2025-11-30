"""Security analysis agent"""

from typing import List, Dict, Any
import re
import logging

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings

logger = logging.getLogger(__name__)


class SecurityAgent:
    """Agent specialized in security vulnerability detection"""
    
    def __init__(self):
        """Initialize security agent"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert security analyst specializing in code security.
            Analyze the provided code for security vulnerabilities including:
            - SQL injection
            - XSS (Cross-Site Scripting)
            - CSRF (Cross-Site Request Forgery)
            - Authentication/Authorization issues
            - Sensitive data exposure
            - Insecure dependencies
            - Input validation issues
            
            For each finding, provide:
            1. Severity (critical/high/medium/low)
            2. Description
            3. Line numbers (if applicable)
            4. Recommendation for fix
            5. CWE/CVE reference if applicable
            
            Return findings as JSON array."""),
            ("user", "Analyze this {language} code for security vulnerabilities:\n\n{code}")
        ])
    
    async def analyze(
        self,
        code: str,
        language: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze code for security vulnerabilities
        
        Args:
            code: Source code to analyze
            language: Programming language
            context: Additional context
            
        Returns:
            List of security findings
        """
        logger.info("Running security analysis")
        
        # Run static analysis checks
        static_findings = self._static_analysis(code, language)
        
        # Run LLM-based analysis
        llm_findings = await self._llm_analysis(code, language)
        
        # Combine and deduplicate findings
        all_findings = static_findings + llm_findings
        
        return all_findings
    
    def _static_analysis(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Run rule-based static analysis"""
        findings = []
        
        # SQL injection patterns
        sql_patterns = [
            r'execute\s*\(.*%.*\)',
            r'\.format\s*\(',
            r'\+.*SELECT.*FROM'
        ]
        
        for i, line in enumerate(code.split('\n'), 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "severity": "high",
                        "category": "sql_injection",
                        "line": i,
                        "description": "Potential SQL injection vulnerability",
                        "recommendation": "Use parameterized queries or ORM",
                        "cwe": "CWE-89"
                    })
        
        # Hardcoded credentials
        credential_patterns = [
            r'password\s*=\s*["\'].*["\']',
            r'api_key\s*=\s*["\'].*["\']',
            r'secret\s*=\s*["\'].*["\']'
        ]
        
        for i, line in enumerate(code.split('\n'), 1):
            for pattern in credential_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "severity": "critical",
                        "category": "hardcoded_credentials",
                        "line": i,
                        "description": "Hardcoded credentials detected",
                        "recommendation": "Use environment variables or secret management",
                        "cwe": "CWE-798"
                    })
        
        return findings
    
    async def _llm_analysis(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Run LLM-based security analysis"""
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({
                "code": code,
                "language": language
            })
            
            # Parse LLM response (simplified)
            # In production, implement proper JSON parsing
            return []
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return []
    
    def generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Generate summary of security findings"""
        if not findings:
            return "No security vulnerabilities detected."
        
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        high = sum(1 for f in findings if f.get("severity") == "high")
        medium = sum(1 for f in findings if f.get("severity") == "medium")
        low = sum(1 for f in findings if f.get("severity") == "low")
        
        return f"Found {len(findings)} security issues: {critical} critical, {high} high, {medium} medium, {low} low severity."