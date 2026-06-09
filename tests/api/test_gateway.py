import pytest
from fastapi.testclient import TestClient


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_extract_document_image(client, sample_image):
    """Test document extraction with image"""
    response = client.post(
        "/api/v1/extract",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "pipeline_used" in data
    assert "confidence" in data
    assert "processing_time" in data


def test_extract_document_pdf(client, sample_pdf):
    """Test document extraction with PDF"""
    response = client.post(
        "/api/v1/extract",
        files={"file": ("test.pdf", sample_pdf, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data


def test_extract_no_file(client):
    """Test extraction without file"""
    response = client.post("/api/v1/extract")
    assert response.status_code == 400


def test_extract_invalid_file_type(client):
    """Test extraction with invalid file type"""
    response = client.post(
        "/api/v1/extract",
        files={"file": ("test.txt", b"invalid content", "text/plain")}
    )
    assert response.status_code == 400


def test_list_models(client):
    """Test listing models"""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


def test_get_pipelines(client):
    """Test getting pipelines"""
    response = client.get("/api/v1/pipelines")
    assert response.status_code == 200
    data = response.json()
    assert "pipelines" in data
    assert "ocr" in data["pipelines"]
    assert "vlm" in data["pipelines"]


def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"