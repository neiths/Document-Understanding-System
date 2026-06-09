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

# Download models if not present
if [ ! -d "/app/models/layout_detection" ]; then
    echo "Downloading layout detection model..."
    python scripts/download_models.py layout_detection
fi

if [ ! -d "/app/models/text_detection" ]; then
    echo "Downloading text detection model..."
    python scripts/download_models.py text_detection
fi

if [ ! -d "/app/models/text_recognition" ]; then
    echo "Downloading text recognition model..."
    python scripts/download_models.py text_recognition
fi

if [ ! -d "/app/models/kie" ]; then
    echo "Downloading KIE model..."
    python scripts/download_models.py kie
fi

# Start the OCR pipeline
echo "Starting OCR pipeline..."
exec "$@"