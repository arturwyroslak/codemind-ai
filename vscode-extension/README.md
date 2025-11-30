# CodeMind AI - VS Code Extension

AI-powered code analysis and suggestions directly in Visual Studio Code.

## Features

- 🔍 **Real-time Code Analysis**: Analyze code selections or entire files
- 🤖 **AI-Powered Suggestions**: Get intelligent recommendations from swarm agents
- 🔒 **Security Scanning**: Detect vulnerabilities, SQL injection, XSS
- ⚡ **Performance Insights**: Identify performance bottlenecks and optimization opportunities
- 🏗️ **Architecture Review**: Check SOLID principles and design patterns
- 💡 **Quick Fixes**: Apply AI-suggested fixes with one click
- 📊 **Inline Diagnostics**: See issues directly in your code with underlines

## Installation

### From VSIX (Local)

1. Download the `.vsix` file
2. Open VS Code
3. Go to Extensions (`Ctrl+Shift+X`)
4. Click `...` menu → Install from VSIX
5. Select the downloaded file

### From Marketplace (Coming Soon)

Search for "CodeMind AI" in VS Code Extensions.

## Configuration

1. Open Settings (`Ctrl+,`)
2. Search for "CodeMind"
3. Configure:

```json
{
    "codemind.apiUrl": "https://api.codemind.ai",
    "codemind.apiToken": "your-jwt-token-here",
    "codemind.analysisType": "full",
    "codemind.enableInlineDiagnostics": true,
    "codemind.autoAnalyzeOnSave": false
}
```

### Getting API Token

1. Visit [CodeMind AI Dashboard](https://codemind.ai/dashboard)
2. Sign up or log in
3. Go to API Settings
4. Generate a new API token
5. Copy and paste into VS Code settings

## Usage

### Analyze Selection

1. Select code in editor
2. Right-click → "CodeMind: Analyze Selection"
3. Or use Command Palette (`Ctrl+Shift+P`) → "CodeMind: Analyze Selection"

### Analyze File

1. Open a file
2. Command Palette → "CodeMind: Analyze Current File"
3. Or right-click file in Explorer → "CodeMind: Analyze Current File"

### Apply Fixes

1. Hover over diagnostic (underlined code)
2. Click "Quick Fix" lightbulb
3. Select "CodeMind: Apply Suggested Fix"

### Auto-Analyze on Save

Enable in settings:
```json
{
    "codemind.autoAnalyzeOnSave": true
}
```

## Analysis Types

- **security**: Focus on security vulnerabilities
- **performance**: Focus on performance issues
- **architecture**: Focus on code structure and design
- **full**: Comprehensive analysis (default)

## Supported Languages

- Python
- JavaScript / TypeScript
- Java
- C / C++
- Go
- Rust
- More coming soon!

## Troubleshooting

### Authentication Failed

- Check that your API token is correct
- Ensure token hasn't expired
- Regenerate token if needed

### Network Error

- Verify `codemind.apiUrl` is correct
- Check firewall/proxy settings
- For local development: `http://localhost:8000`

### No Results

- Ensure you have an active editor with code
- Check that the language is supported
- Review Output panel (View → Output → CodeMind AI)

## Privacy

CodeMind AI automatically masks:
- API keys
- Passwords
- Tokens
- PII data

Before sending to analysis servers.

## Feedback

Found a bug or have a suggestion?

- [GitHub Issues](https://github.com/arturwyroslak/codemind-ai/issues)
- [Discord Community](https://discord.gg/codemind)
- Email: support@codemind.ai

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with ❤️ by the CodeMind AI Team**