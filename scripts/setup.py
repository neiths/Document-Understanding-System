#!/usr/bin/env python3
"""
Setup script for Document Understanding System
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors"""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True
    )
    return result

def check_docker():
    """Check if Docker is installed"""
    try:
        result = run_command("docker --version", check=True)
        return True
    except:
        return False

def check_docker_compose():
    """Check if Docker Compose is installed"""
    try:
        result = run_command("docker-compose --version", check=True)
        return True
    except:
        return False

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    run_command("pip install -r requirements.txt")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/annotations",
        "models",
        "logs",
        "temp_uploads",
        "config/ssl",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_pretrained_models():
    """Download pretrained models"""
    print("Downloading pretrained models...")
    run_command("python scripts/download_models.py layout_detection")
    run_command("python scripts/download_models.py text_detection")
    run_command("python scripts/download_models.py text_recognition")
    run_command("python scripts/download_models.py kie")

def setup_mlflow():
    """Initialize MLflow database"""
    print("Setting up MLflow...")
    run_command("docker run -d -p 5000:5000 -e AWS_ACCESS_KEY_ID=minioadmin -e AWS_SECRET_ACCESS_KEY=minioadmin -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 --name mlflow mlflow/mlflow:latest")

def setup_minio():
    """Initialize MinIO"""
    print("Setting up MinIO...")
    run_command("docker run -d -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -v minio_data:/data --name minio minio/minio:latest server /data --console-address :9001")

def start_system():
    """Start the entire system"""
    print("Starting Document Understanding System...")
    run_command("docker-compose up -d")

def run_tests():
    """Run all tests"""
    print("Running tests...")
    run_command("pytest tests/ -v")

def main():
    parser = argparse.ArgumentParser(description="Setup Document Understanding System")
    parser.add_argument("--check-requirements", action="store_true", help="Check system requirements")
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument("--create-dirs", action="store_true", help="Create necessary directories")
    parser.add_argument("--download-models", action="store_true", help="Download pretrained models")
    parser.add_argument("--setup-infrastructure", action="store_true", help="Setup MLflow and MinIO")
    parser.add_argument("--start", action="store_true", help="Start the entire system")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--full-setup", action="store_true", help="Run full setup process")

    args = parser.parse_args()

    if args.check_requirements:
        print("Checking system requirements...")

        if not check_docker():
            print("❌ Docker is not installed. Please install Docker first.")
            sys.exit(1)

        if not check_docker_compose():
            print("❌ Docker Compose is not installed. Please install Docker Compose first.")
            sys.exit(1)

        gpu_available = check_gpu()
        print(f"✅ Docker: Installed")
        print(f"✅ Docker Compose: Installed")
        print(f"✅ GPU: {'Available' if gpu_available else 'Not available'}")

        if not gpu_available:
            print("⚠️  GPU not detected. The system will run on CPU, but performance will be slower.")
            print("   Consider installing CUDA/cuDNN for better performance.")

    if args.full_setup or args.install_deps:
        install_python_dependencies()

    if args.full_setup or args.create_dirs:
        create_directories()

    if args.full_setup or args.download_models:
        download_pretrained_models()

    if args.full_setup or args.setup_infrastructure:
        setup_mlflow()
        setup_minio()

    if args.start:
        start_system()

    if args.test:
        run_tests()

    if args.full_setup:
        print("\n🎉 Full setup completed!")
        print("\nThe Document Understanding System is ready to use.")
        print("\nAccess the services at:")
        print("- API Gateway: http://localhost")
        print("- MLflow UI: http://localhost:5000")
        print("- MinIO Console: http://localhost:9001")
        print("- Grafana: http://localhost:3000 (admin/admin)")
        print("- Prometheus: http://localhost:9090")
        print("\nTo test the API, run:")
        print("curl -X POST \"http://localhost:8000/api/v1/extract\" -F \"file=@path/to/your/document.pdf\"")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()