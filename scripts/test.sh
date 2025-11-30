#!/bin/bash
# Run tests for CodeMind AI

set -e

echo "🧪 Running tests..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run backend tests
echo "🔍 Running backend tests..."
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
cd ..

# Run frontend tests (if exists)
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo "🔍 Running frontend tests..."
    cd frontend
    if grep -q '"test"' package.json; then
        npm test -- --watchAll=false
    fi
    cd ..
fi

echo "✅ All tests passed!"