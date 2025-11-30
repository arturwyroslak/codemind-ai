# 🧠 CodeMind AI

**Intelligent Code Analysis, Refactoring and Generation Platform**

CodeMind AI is an advanced platform that leverages Multi-Agent Systems, RAG (Retrieval-Augmented Generation), and Natural Language Processing to revolutionize software development.

## ✨ Features

### 🤖 Multi-Agent Code Analysis
- **Security Agent** - Identifies vulnerabilities and security issues
- **Performance Agent** - Analyzes performance bottlenecks and optimization opportunities
- **Architecture Agent** - Reviews code structure and design patterns
- **Documentation Agent** - Generates comprehensive documentation

### 📚 RAG-Enhanced Documentation
- Automatic context-aware documentation generation
- Code repository indexing with vector embeddings
- Semantic search across codebase

### 💬 Natural Language to Code
- Convert natural language descriptions to working code
- Support for multiple programming languages
- Context-aware code generation based on project structure

### 👥 Real-time Collaboration
- Live collaboration between developers and AI agents
- Interactive code review sessions
- Instant feedback and suggestions

### 📊 Code Quality Dashboard
- Visual metrics and analytics
- Historical trend analysis
- Customizable quality gates

## 🛠️ Technology Stack

### Backend
- **Python 3.11+**
- **FastAPI** - High-performance web framework
- **LangGraph** - Agent orchestration
- **LangChain** - LLM framework

### Frontend
- **React 18** with TypeScript
- **Vite** - Build tool
- **Monaco Editor** - Code editor
- **TailwindCSS** - Styling

### AI & ML
- **OpenAI GPT-4** - Primary LLM
- **Anthropic Claude** - Alternative LLM
- **Ollama** - Local LLM support
- **Pinecone** - Vector database

### DevOps
- **Docker** & **Docker Compose**
- **Prometheus** - Metrics
- **Grafana** - Visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/arturwyroslak/codemind-ai.git
cd codemind-ai
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start with Docker Compose**
```bash
docker-compose up -d
```

4. **Or run locally**

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000

## 📚 Documentation

### API Endpoints

#### Code Analysis
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "code": "your code here",
  "language": "python",
  "agents": ["security", "performance", "architecture"]
}
```

#### Generate Code
```http
POST /api/v1/generate
Content-Type: application/json

{
  "description": "Create a FastAPI endpoint for user authentication",
  "language": "python",
  "framework": "fastapi"
}
```

#### RAG Query
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "How does authentication work in this project?",
  "repository_id": "repo-123"
}
```

## 🏭 Architecture

```
┌────────────────────────┐
│   React Frontend         │
│   (TypeScript + Vite)    │
└───────────┬────────────┘
            │
            │ REST API
            │
┌───────────┴────────────┐
│   FastAPI Gateway        │
│   (Python)               │
└───────────┬────────────┘
            │
    ┌───────┼───────┐
    │       │       │
┌───┴───┐ ┌─┴──┐ ┌─┴───┐
│ Agent │ │RAG│ │ LLM │
│ Graph │ │   │ │     │
└───────┘ └───┘ └─────┘
    │       │       │
    │   ┌───┴───┐   │
    │   │ Vector│   │
    │   │  DB   │   │
    │   └───────┘   │
    │               │
┌───┴───────────────┴───┐
│    Metrics & Logs       │
│ (Prometheus/Grafana)  │
└────────────────────────┘
```

## 🤖 Agent System

CodeMind AI uses a sophisticated multi-agent system built with LangGraph:

### Agent Types

1. **Security Agent**
   - SQL injection detection
   - XSS vulnerability scanning
   - Authentication/authorization issues
   - Dependency vulnerability checking

2. **Performance Agent**
   - Algorithm complexity analysis
   - Database query optimization
   - Memory leak detection
   - Caching opportunities

3. **Architecture Agent**
   - Design pattern recognition
   - SOLID principles validation
   - Code smell detection
   - Refactoring suggestions

4. **Documentation Agent**
   - Docstring generation
   - API documentation
   - Architecture diagrams
   - Usage examples

## 📊 Monitoring & Metrics

- Request/response times
- Agent execution metrics
- LLM token usage
- Error rates
- User activity

## 🧑‍💻 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 Roadmap

- [ ] Support for more programming languages
- [ ] IDE plugins (VSCode, JetBrains)
- [ ] Team collaboration features
- [ ] Custom agent creation
- [ ] Advanced code refactoring tools
- [ ] Integration with CI/CD pipelines
- [ ] Mobile app

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/arturwyroslak/codemind-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/arturwyroslak/codemind-ai/discussions)

## 🙏 Acknowledgments

Inspired by cutting-edge projects:
- Microsoft Generative AI for Beginners
- Genesis Embodied AI
- RAG Techniques by NirDiamant
- GenAI Agents by NirDiamant
- WrenAI by Canner

---

**Built with ❤️ by the CodeMind AI Team**