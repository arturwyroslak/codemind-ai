#!/bin/bash
# Setup script for CodeMind AI

set -e

echo "🚀 Setting up CodeMind AI..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo "📥 Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install frontend dependencies
if command -v npm &> /dev/null; then
    echo "📥 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
else
    echo "⚠️  npm not found, skipping frontend setup"
fi

# Install VS Code extension dependencies
if command -v npm &> /dev/null; then
    echo "📥 Installing VS Code extension dependencies..."
    cd vscode-extension
    npm install
    cd ..
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run 'docker-compose up -d' to start services"
echo "3. Run 'cd backend && uvicorn app.main:app --reload' to start backend"
echo "4. Run 'cd frontend && npm start' to start frontend"
echo ""
echo "Happy coding! 🎉"