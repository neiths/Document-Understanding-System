# Deployment Guide

This guide covers how to deploy the Document Understanding System in various environments.

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM (16GB recommended)
- GPU with at least 8GB VRAM (optional, but recommended)
- 20GB free disk space

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd document-understanding-system
```

### 2. Run the Setup Script

```bash
python scripts/setup.py --full-setup
```

This will:
- Install Python dependencies
- Create necessary directories
- Download pretrained models
- Setup MLflow and MinIO
- Start the entire system

### 3. Verify Services

Check if all services are running:

```bash
docker-compose ps
```

## Manual Deployment

### 1. Environment Setup

```bash
# Create environment file
cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model Storage
MLFLOW_TRACKING_URI=http://mlflow:5000
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=model-registry

# Model Configuration
VLM_MODEL=qwen2.5-vl-7b
ENABLE_OLLAMA=true

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF
```

### 2. Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 3. Access Services

- **API Gateway**: http://localhost
- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Configuration

### Custom Configuration

Edit `docker-compose.yml` to adjust resources:

```yaml
services:
  ocr-pipeline:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              # For specific GPU
              device_ids: ['0']

  vlm-pipeline:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              # Memory limit
              capabilities: ['gpu']
              limits:
                memory: 16G
```

### Environment Variables

Create `.env` file for custom configuration:

```bash
# Gateway Configuration
GATEWAY_WORKERS=4
GATEWAY_TIMEOUT=300

# OCR Pipeline
OCR_BATCH_SIZE=8
OCR_CONFIDENCE_THRESHOLD=0.7

# VLM Pipeline
VLM_MAX_TOKENS=4096
VLM_TEMPERATURE=0.1

# Monitoring
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

## Scaling

### Horizontal Scaling

```bash
# Scale API gateway
docker-compose up -d --scale fastapi-gateway=3

# Scale pipelines
docker-compose up -d --scale ocr-pipeline=2
docker-compose up -d --scale vlm-pipeline=2
```

### Vertical Scaling

Adjust resource limits in `docker-compose.yml`:

```yaml
services:
  ocr-pipeline:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
```

## Monitoring and Logging

### View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f ocr-pipeline

# Follow logs with timestamp
docker-compose logs -f --tail=100 ocr-pipeline
```

### Metrics Access

```bash
# Access Prometheus
curl http://localhost:9090/api/v1/query?query=document_requests_total

# Access Grafana dashboard
open http://localhost:3000
```

### Health Checks

```bash
# Check gateway health
curl http://localhost/health

# Check pipeline health
curl http://localhost:8001/health
```

## Production Deployment

### 1. Security Configuration

```yaml
# nginx.conf - Add SSL
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_session_cache shared:SSL:1m;
    ssl_session_timeout 5m;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
}
```

### 2. Kubernetes Deployment

Create `k8s/` directory with deployment manifests:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-gateway
  template:
    metadata:
      labels:
        app: document-gateway
    spec:
      containers:
      - name: gateway
        image: your-registry/document-gateway:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### 3. Load Balancing

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: document-gateway-service
spec:
  selector:
    app: document-gateway
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Reduce batch size
   export OCR_BATCH_SIZE=4
   ```

2. **Model Loading Errors**
   ```bash
   # Check model downloads
   ls -la models/
   
   # Redownload models
   python scripts/download_models.py --all
   ```

3. **Memory Issues**
   ```bash
   # Check container memory
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

### Debug Mode

```bash
# Run with debug logging
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d

# Enable debug mode
export DEBUG=true
```

### Performance Tuning

1. **GPU Optimization**
   - Use FP16 when available
   - Optimize batch sizes
   - Use TensorRT if available

2. **CPU Optimization**
   - Enable CPU pinning
   - Use multiple workers
   - Optimize Docker network settings

3. **Model Optimization**
   - Quantize models
   - Use model pruning
   - Cache frequently used models

## Backup and Recovery

### Backup Models

```bash
# Backup to MinIO
mc alias set minio http://localhost:9000 minioadmin minioadmin
mc cp models/ minio/models-backup/

# Backup MLflow
docker run --rm -v mlflow_data:/data \
  alpine tar czf - -C /data . | gzip > mlflow-backup.tar.gz
```

### Restore from Backup

```bash
# Restore models
mc cp minio/models-backup/ models/

# Restore MLflow
gunzip < mlflow-backup.tar.gz | docker run --rm -i -v mlflow_data:/data alpine tar xzf - -C /data
```