#!/bin/bash
# Deployment script for CodeMind AI

set -e

echo "🚀 Deploying CodeMind AI..."

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose -f docker-compose.advanced.yml build

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.advanced.yml down

# Start services
echo "▶️  Starting services..."
docker-compose -f docker-compose.advanced.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check health
echo "🏥 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
    exit 1
fi

echo "✅ Deployment complete!"
echo "📊 Services running:"
docker-compose -f docker-compose.advanced.yml ps