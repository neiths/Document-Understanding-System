from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
import logging
from typing import Dict, Any
import json
from pathlib import Path
import os

# Import our modules
from ..common.config import settings
from ..common.models import ExtractionRequest, ExtractionResponse, DocumentType
from ..common.utils import save_upload_file, get_file_info, validate_file_type
from ..common.ml_client import MLflowClient, MinIOClient
from ..router.document_router import DocumentRouter
from ..monitoring.metrics import MetricsCollector

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Understanding System Gateway",
    description="API gateway for document understanding with dual-pipeline architecture",
    version="1.0.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
mlflow_client = MLflowClient(
    tracking_uri=settings.mlflow_tracking_uri,
    artifact_uri=f"http://{settings.minio_endpoint}/{settings.minio_bucket}"
)

minio_client = MinIOClient(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    bucket=settings.minio_bucket
)

# Initialize router
document_router = DocumentRouter(
    mlflow_client=mlflow_client,
    minio_client=minio_client,
    config=settings.routing_config
)

# Initialize metrics
metrics = MetricsCollector()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'document_requests_total',
    'Total number of document processing requests',
    ['pipeline', 'document_type', 'status']
)

REQUEST_DURATION = Histogram(
    'document_processing_seconds',
    'Time spent processing documents',
    ['pipeline']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests'
)


@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = None
):
    """Extract text from uploaded document"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Check file type
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        temp_file_path = f"temp_uploads/{file.filename}"

        if not validate_file_type(temp_file_path, allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF, JPG, PNG, TIFF, or BMP."
            )

        # Save file
        file_path = save_upload_file(file, temp_file_path)
        file_info = get_file_info(file_path)

        # Create extraction request
        doc_type = DocumentType(document_type) if document_type else None
        request = ExtractionRequest(
            file_path=file_path,
            document_type=doc_type,
            metadata={"file_size": file_info['size'], "file_type": file_info['extension']}
        )

        # Route to appropriate pipeline
        result = await document_router.route_and_process(request)

        # Update metrics
        processing_time = time.time() - start_time
        REQUEST_COUNT.labels(
            pipeline=result.pipeline_used,
            document_type=result.data.get('document_type', 'unknown') if result.data else 'unknown',
            status='success' if result.success else 'error'
        ).inc()

        REQUEST_DURATION.labels(pipeline=result.pipeline_used).observe(processing_time)

        return ExtractionResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            pipeline_used=result.pipeline_used,
            confidence=result.confidence,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        REQUEST_COUNT.labels(
            pipeline='unknown',
            document_type='unknown',
            status='error'
        ).inc()

        return ExtractionResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )

    finally:
        ACTIVE_REQUESTS.dec()

        # Clean up temp file in background
        background_tasks.add_task(cleanup_temp_files, [file_path])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)


@app.get("/api/v1/models")
async def list_models():
    """List available models"""
    try:
        models = minio_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not list models")


@app.get("/api/v1/pipelines")
async def get_pipelines():
    """Get available pipelines and their status"""
    return {
        "pipelines": {
            "ocr": {
                "status": "active",
                "description": "High confidence OCR pipeline for standard forms"
            },
            "vlm": {
                "status": "active",
                "description": "Vision-Language Model pipeline for complex layouts"
            }
        }
    }


def cleanup_temp_files(file_paths: list):
    """Clean up temporary files"""
    from ..common.utils import cleanup_temp_files
    cleanup_temp_files(file_paths)


if __name__ == "__main__":
    import uvicorn

    # Create temp uploads directory
    os.makedirs("temp_uploads", exist_ok=True)

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )