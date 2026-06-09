import pytest
import os
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock

# Import the FastAPI app
from src.gateway.main import app

# Configure test environment
os.environ["TESTING"] = "true"


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    from PIL import Image
    import io

    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='white')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    return image_bytes


@pytest.fixture
def sample_pdf():
    """Create a sample PDF for testing"""
    from PyPDF2 import PdfWriter
    import io

    # Create a simple test PDF
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page.merge_page(writer.pages[0])

    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)

    return pdf_bytes


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client"""
    mock = Mock()
    mock.get_latest_model.return_value = "models:/DocLayout-YOLO/1"
    mock.load_model.return_value = Mock()
    mock.log_model_metrics.return_value = None
    return mock


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client"""
    mock = Mock()
    mock.list_models.return_value = {}
    mock.download_model.return_value = None
    return mock