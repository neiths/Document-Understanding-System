#!/bin/bash

# Start script for Document Understanding System

set -e

echo "Starting Document Understanding System..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models logs temp_uploads

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check health
echo "Checking service health..."
curl -f http://localhost/health > /dev/null 2>&1 && echo "✅ Gateway is healthy" || echo "❌ Gateway is not healthy"

# Access information
echo ""
echo "🎉 System started successfully!"
echo ""
echo "Access the services at:"
echo "- API Gateway: http://localhost"
echo "- MLflow UI: http://localhost:5000"
echo "- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- Prometheus: http://localhost:9090"
echo ""
echo "To test the API:"
echo "curl -X POST \"http://localhost:8000/api/v1/extract\" -F \"file=@path/to/your/document.pdf\""
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"