from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model Storage
    mlflow_tracking_uri: str = "http://localhost:5000"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "model-registry"

    # Model Configuration
    ocr_models = {
        "layout_detection": {
            "model_name": "DocLayout-YOLO",
            "version": "v1.0",
            "confidence_threshold": 0.8,
        },
        "text_detection": {
            "model_name": "DBNet",
            "version": "v1.0",
            "confidence_threshold": 0.7,
        },
        "text_recognition": {
            "model_name": "PARSeq",
            "version": "v1.0",
            "confidence_threshold": 0.8,
        },
        "kie": {
            "model_name": "LayoutLMv3",
            "version": "v1.0",
            "confidence_threshold": 0.75,
        },
    }

    vlm_models = {
        "qwen2.5-vl": {
            "model_name": "qwen2.5-vl-7b",
            "version": "v1.0",
            "max_tokens": 4096,
            "temperature": 0.1,
        }
    }

    # Routing Configuration
    routing_config = {
        "confidence_threshold": 0.7,
        "use_ocr_for": ["standard", "form", "receipt"],
        "use_vlm_for": ["complex", "unstructured", "multi-page"],
    }

    # Monitoring Configuration
    prometheus_port: int = 8001
    enable_metrics: bool = True

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    class Config:
        env_file = ".env"


settings = Settings()
