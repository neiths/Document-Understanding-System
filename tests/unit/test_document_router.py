import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.router.document_router import DocumentRouter, DocumentClassifier


@pytest.fixture
def mock_mlflow_client():
    return Mock()


@pytest.fixture
def mock_minio_client():
    return Mock()


@pytest.fixture
def document_router(mock_mlflow_client, mock_minio_client):
    return DocumentRouter(
        mlflow_client=mock_mlflow_client,
        minio_client=mock_minio_client,
        config={"confidence_threshold": 0.7},
    )


@pytest.fixture
def document_classifier():
    return DocumentClassifier()


def test_document_router_initialization(document_router):
    """Test document router initialization"""
    assert document_router.ocr_pipeline is not None
    assert document_router.vlm_pipeline is not None
    assert document_router.document_classifier is not None


def test_determine_route_with_user_preference(document_router):
    """Test route determination with user preference"""
    from src.common.models import DocumentType

    # User prefers OCR
    pipeline = document_router._determine_pipeline(
        Mock(document_type=DocumentType.STANDARD), DocumentType.STANDARD
    )
    assert pipeline == "ocr"

    # User prefers VLM
    pipeline = document_router._determine_pipeline(
        Mock(document_type=DocumentType.COMPLEX), DocumentType.COMPLEX
    )
    assert pipeline == "vlm"


def test_determine_route_with_classification(document_router):
    """Test route determination based on classification"""
    high_confidence = Mock(document_type="standard", confidence=0.8)
    low_confidence = Mock(document_type="complex", confidence=0.6)

    # High confidence -> OCR
    pipeline = document_router._determine_pipeline(high_confidence, None)
    assert pipeline == "ocr"

    # Low confidence -> VLM
    pipeline = document_router._determine_pipeline(low_confidence, None)
    assert pipeline == "vlm"


@pytest.mark.asyncio
async def test_route_and_process_success(document_router, sample_image):
    """Test successful routing and processing"""
    from src.common.models import ExtractionRequest

    # Mock successful processing from both pipelines
    document_router.ocr_pipeline.process = Mock(
        return_value=Mock(success=True, pipeline_used="ocr", confidence=0.8)
    )

    request = ExtractionRequest(file_path="test.jpg")
    result = await document_router.route_and_process(request)

    assert result.success is True
    assert result.pipeline_used == "ocr"


@pytest.mark.asyncio
async def test_route_and_process_error(document_router):
    """Test error handling in routing"""
    from src.common.models import ExtractionRequest

    # Mock error from OCR pipeline
    document_router.ocr_pipeline.process = Mock(side_effect=Exception("Test error"))

    request = ExtractionRequest(file_path="test.jpg")
    result = await document_router.route_and_process(request)

    assert result.success is False
    assert result.error is not None


def test_document_classifier_initialization(document_classifier):
    """Test document classifier initialization"""
    assert document_classifier.type_patterns is not None
    assert "form" in document_classifier.type_patterns
    assert "receipt" in document_classifier.type_patterns
    assert "standard" in document_classifier.type_patterns


@pytest.mark.asyncio
async def test_classify_document_form(document_classifier):
    """Test document classification for form"""
    # Create a mock image path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"fake image content")
        tmp_path = tmp.name

    try:
        result = await document_classifier.classify(tmp_path)

        assert "document_type" in result
        assert "confidence" in result
        assert "scores" in result
        assert "file_info" in result

    finally:
        # Clean up
        os.unlink(tmp_path)


def test_analyze_file(document_classifier):
    """Test file analysis"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        # Create a minimal PDF
        from PyPDF2 import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(612, 792)
        writer.write(tmp)
        tmp_path = tmp.name

        result = document_classifier._analyze_file(tmp_path)

        assert "size" in result
        assert "extension" in result
        assert "pages" in result
        assert result["extension"] == ".pdf"


def test_extract_text_sample(document_classifier):
    """Test text sample extraction"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"fake image content")
        tmp_path = tmp.name

    try:
        with patch("src.router.document_router.perform_ocr") as mock_ocr:
            mock_ocr.return_value = {"text": "Test document text", "boxes": []}

            result = document_classifier._extract_text_sample(tmp_path)
            assert result == "test document text"

    finally:
        os.unlink(tmp_path)


def test_calculate_score(document_classifier):
    """Test classification score calculation"""
    text = "This is a form with name and address fields"
    file_info = {"pages": 1, "extension": ".pdf"}

    patterns = document_classifier.type_patterns["form"]
    score = document_classifier._calculate_score(text, patterns, file_info)

    assert isinstance(score, float)
    assert 0 <= score <= 1
