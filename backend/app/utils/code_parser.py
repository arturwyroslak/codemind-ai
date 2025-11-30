import re
from typing import Dict, List, Optional
import ast

class CodeParser:
    """Parse and analyze code structure."""
    
    @staticmethod
    def parse_python(code: str) -> Dict:
        """Parse Python code and extract structure."""
        try:
            tree = ast.parse(code)
            return {
                "functions": CodeParser._extract_functions(tree),
                "classes": CodeParser._extract_classes(tree),
                "imports": CodeParser._extract_imports(tree),
                "complexity": CodeParser._estimate_complexity(tree)
            }
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
    
    @staticmethod
    def _extract_functions(tree) -> List[str]:
        """Extract function names from AST."""
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    @staticmethod
    def _extract_classes(tree) -> List[str]:
        """Extract class names from AST."""
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    @staticmethod
    def _extract_imports(tree) -> List[str]:
        """Extract import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    @staticmethod
    def _estimate_complexity(tree) -> int:
        """Estimate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    @staticmethod
    def extract_docstrings(code: str) -> Dict[str, str]:
        """Extract docstrings from Python code."""
        try:
            tree = ast.parse(code)
            docstrings = {}
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        name = getattr(node, 'name', 'module')
                        docstrings[name] = docstring
            
            return docstrings
        except:
            return {}
    
    @staticmethod
    def count_lines(code: str) -> Dict[str, int]:
        """Count lines of code (total, code, comments, blank)."""
        lines = code.split('\n')
        total = len(lines)
        blank = sum(1 for line in lines if not line.strip())
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total - blank - comments
        
        return {
            "total": total,
            "code": code_lines,
            "comments": comments,
            "blank": blank
        }