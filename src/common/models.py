from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    STANDARD = "standard"
    FORM = "form"
    RECEIPT = "receipt"
    COMPLEX = "complex"
    UNSTRUCTURED = "unstructured"
    MULTI_PAGE = "multi-page"


class ExtractionRequest(BaseModel):
    file_path: str
    document_type: Optional[DocumentType] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LayoutDetectionResult(BaseModel):
    boxes: List[Dict[str, Any]]
    confidence: float
    document_type: DocumentType


class TextDetectionResult(BaseModel):
    text_blocks: List[Dict[str, Any]]
    confidence: float


class TextRecognitionResult(BaseModel):
    text: str
    confidence: float
    bbox: Dict[str, Any]


class KIEResult(BaseModel):
    key_value_pairs: Dict[str, Any]
    confidence: float
    entities: List[Dict[str, Any]]


class OCRPipelineResult(BaseModel):
    layout_detection: LayoutDetectionResult
    text_detection: TextDetectionResult
    text_recognition: List[TextRecognitionResult]
    kie: KIEResult
    pipeline_confidence: float
    processing_time: float


class VLMResponse(BaseModel):
    extracted_text: str
    structured_data: Dict[str, Any]
    confidence: float
    processing_time: float
    reasoning: Optional[str] = None


class ExtractionResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    pipeline_used: str
    confidence: float
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    model_name: str
    version: str
    size: Optional[int] = None
    download_url: Optional[str] = None
    local_path: Optional[str] = None
    is_available: bool = False


class MetricsData(BaseModel):
    request_count: int = 0
    ocr_pipeline_count: int = 0
    vlm_pipeline_count: int = 0
    average_confidence: float = 0.0
    average_ocr_time: float = 0.0
    average_vlm_time: float = 0.0
    error_count: int = 0
