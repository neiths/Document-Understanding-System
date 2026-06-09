# Document Understanding System

A comprehensive document understanding system with dual-pipeline architecture (OCR + VLM) for processing various document types.

## Architecture Overview

The system consists of 5 main phases:

### Phase 1: Core Serving & Inference Layer (Block 4)
- Gateway & Routing with NGINX and FastAPI
- Dual Pipeline Execution (OCR & VLM)
- Document classification and routing logic

### Phase 2: Model Registry & Storage (Block 2)
- MinIO/S3 for model artifact storage
- MLflow Model Registry for version management

### Phase 3: Monitoring & Observability (Block 5)
- Prometheus metrics collection
- Grafana dashboards
- Loki log aggregation
- OpenTelemetry tracing

### Phase 4: CI/CD Pipeline (Block 3)
- GitHub Actions workflow
- Docker containerization
- Container registry deployment

### Phase 5: Data Preparation & Model Training (Block 1)
- Dataset aggregation
- Annotation with Label Studio
- Model training and evaluation

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Start services: `docker-compose up -d`
3. Test API: `curl -X POST "http://localhost:8000/api/v1/extract" -F "file=@document.pdf"`

## API Endpoints

- `POST /api/v1/extract` - Extract text from document
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics