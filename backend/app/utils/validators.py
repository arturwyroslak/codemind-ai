import re
from typing import Optional

class Validators:
    """Input validation utilities."""
    
    @staticmethod
    def validate_repository(repo: str) -> bool:
        """Validate repository format (owner/repo)."""
        pattern = r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$'
        return bool(re.match(pattern, repo))
    
    @staticmethod
    def validate_language(language: str) -> bool:
        """Validate programming language."""
        supported = [
            'python', 'javascript', 'typescript', 'java',
            'cpp', 'c', 'go', 'rust', 'ruby', 'php',
            'swift', 'kotlin', 'scala', 'csharp'
        ]
        return language.lower() in supported
    
    @staticmethod
    def validate_analysis_type(analysis_type: str) -> bool:
        """Validate analysis type."""
        valid_types = ['security', 'performance', 'architecture', 'full']
        return analysis_type.lower() in valid_types
    
    @staticmethod
    def sanitize_code(code: str, max_length: int = 100000) -> str:
        """Sanitize code input."""
        if len(code) > max_length:
            raise ValueError(f"Code exceeds maximum length of {max_length}")
        return code.strip()
    
    @staticmethod
    def validate_github_token(token: str) -> bool:
        """Validate GitHub token format."""
        # GitHub Personal Access Tokens
        if token.startswith('ghp_') and len(token) == 40:
            return True
        # GitHub OAuth tokens
        if token.startswith('gho_'):
            return True
        return False