# Contributing to CodeMind AI

Thank you for your interest in contributing to CodeMind AI! 🎉

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker 20+
- Git

### Quick Start

```bash
# Clone repository
git clone https://github.com/arturwyroslak/codemind-ai.git
cd codemind-ai

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Project Structure

```
codemind-ai/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── agents/    # AI agents
│   │   ├── api/       # API endpoints
│   │   ├── core/      # Core functionality
│   │   └── rag/       # RAG pipeline
│   └── tests/         # Backend tests
├── frontend/          # React frontend
│   └── src/
│       └── components/
├── vscode-extension/  # VS Code extension
│   └── src/
└── scripts/           # Utility scripts
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow PEP 8 for Python code
- Use ESLint rules for TypeScript/JavaScript
- Write tests for new features
- Update documentation

### 3. Run Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Or use script
./scripts/test.sh
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Contributing Areas

### 1. Agent Development

Create new analysis agents:

```python
from app.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def analyze(self, code: str) -> dict:
        # Your analysis logic
        return {"findings": []}
```

### 2. Plugin Creation

Integrate with external tools:

```python
from app.advanced_features import BasePlugin

class CustomPlugin(BasePlugin):
    async def execute(self, task: dict) -> dict:
        # Plugin logic
        return {"status": "success"}
```

### 3. Prompt Engineering

Contribute prompt templates in `prompts/` directory.

### 4. Performance Optimization

- Profile slow functions
- Optimize database queries
- Improve caching strategies

### 5. Documentation

- Improve README
- Add code examples
- Write tutorials

## Code Style

### Python

```python
# Use type hints
def analyze_code(code: str, language: str) -> dict:
    pass

# Docstrings
def function():
    """Brief description.
    
    Args:
        param: Description
    
    Returns:
        Description
    """
```

### TypeScript

```typescript
// Use interfaces
interface AnalysisResult {
  score: number;
  findings: Finding[];
}

// Async/await
async function analyze(): Promise<AnalysisResult> {
  // ...
}
```

## Testing Guidelines

- Write unit tests for new functions
- Write integration tests for API endpoints
- Aim for >80% code coverage
- Use fixtures for common test data

## Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what and why
3. **Tests**: Include test coverage
4. **Documentation**: Update relevant docs
5. **Screenshots**: For UI changes

## Getting Help

- [Discord Community](https://discord.gg/codemind)
- [GitHub Discussions](https://github.com/arturwyroslak/codemind-ai/discussions)
- [Documentation](https://docs.codemind.ai)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow project guidelines

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CodeMind AI! 🚀