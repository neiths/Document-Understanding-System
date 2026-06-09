#!/bin/bash

set -e

# Wait for MLflow and MinIO to be ready
echo "Waiting for MLflow..."
until curl -f http://mlflow:5000/health; do
  echo "Waiting for MLflow..."
  sleep 5
done

echo "Waiting for MinIO..."
until curl -f http://minio:9000/minio/health/live; do
  echo "Waiting for MinIO..."
  sleep 5
done

# Download VLM model if not present
if [ ! -d "/app/models/qwen2.5-vl" ]; then
    echo "Downloading Qwen2.5-VL model..."
    python scripts/download_models.py vlm qwen2.5-vl
fi

# Start Ollama in background if configured
if [ "$ENABLE_OLLAMA" = "true" ]; then
    echo "Starting Ollama..."
    ollama serve &
    # Wait for Ollama to be ready
    until curl -f http://localhost:11434/api/tags; do
      echo "Waiting for Ollama..."
      sleep 5
    done

    # Pull the model if not already pulled
    if ! ollama list | grep -q qwen2.5-vl; then
        echo "Pulling Qwen2.5-VL model..."
        ollama pull qwen2.5-vl
    fi
fi

# Start the VLM pipeline
echo "Starting VLM pipeline..."
exec "$@"