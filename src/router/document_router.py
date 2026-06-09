from typing import Dict, Any, Optional
import logging
from pathlib import Path
import os

from ..common.models import ExtractionRequest, ExtractionResponse, DocumentType
from ..common.ml_client import MLflowClient, MinIOClient
from ..ocr.pipeline import OCRPipeline
from ..vlm.pipeline import VLMPipeline
from ..common.config import Settings

logger = logging.getLogger(__name__)


class DocumentRouter:
    """Router for document processing pipelines"""

    def __init__(
        self,
        mlflow_client: MLflowClient,
        minio_client: MinIOClient,
        config: Dict[str, Any],
    ):
        self.mlflow_client = mlflow_client
        self.minio_client = minio_client
        self.config = config

        # Initialize pipelines
        self.ocr_pipeline = OCRPipeline(mlflow_client, minio_client)
        self.vlm_pipeline = VLMPipeline(mlflow_client, minio_client)

        # Initialize document classifier
        self.document_classifier = DocumentClassifier()

    async def route_and_process(self, request: ExtractionRequest) -> ExtractionResponse:
        """Route document to appropriate pipeline and process"""
        try:
            # Classify document
            classification = await self.document_classifier.classify(request.file_path)

            # Determine pipeline based on classification
            pipeline_type = self._determine_pipeline(
                classification, request.document_type
            )

            logger.info(f"Routing {request.file_path} to {pipeline_type} pipeline")

            # Process through appropriate pipeline
            if pipeline_type == "ocr":
                result = await self.ocr_pipeline.process(request)
            else:
                result = await self.vlm_pipeline.process(request)

            # Add classification metadata
            if result.data:
                result.data.update(
                    {
                        "classification": classification.model_dump(),
                        "pipeline_used": pipeline_type,
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error routing document: {str(e)}")
            return ExtractionResponse(
                success=False, error=str(e), pipeline_used="unknown"
            )

    def _determine_pipeline(
        self, classification, user_requested_type: Optional[DocumentType]
    ) -> str:
        """Determine which pipeline to use based on classification"""
        # If user specifies a type, respect it
        if user_requested_type:
            if user_requested_type in self.config["use_ocr_for"]:
                return "ocr"
            elif user_requested_type in self.config["use_vlm_for"]:
                return "vlm"

        # Use classification confidence
        if classification.confidence >= self.config["confidence_threshold"]:
            # High confidence - use OCR for known types
            if classification.document_type in self.config["use_ocr_for"]:
                return "ocr"

        # Default to VLM for uncertain or complex documents
        return "vlm"


class DocumentClassifier:
    """Simple document classifier using rule-based approach"""

    def __init__(self):
        # Define document type characteristics
        self.type_patterns = {
            DocumentType.FORM: {
                "keywords": ["form", "application", "registration"],
                "structure_indicators": ["table", "grid", "checkbox", "input"],
                "confidence": 0.8,
            },
            DocumentType.RECEIPT: {
                "keywords": ["receipt", "purchase", "transaction", "payment", "store"],
                "structure_indicators": ["total", "date", "item", "price", "quantity"],
                "confidence": 0.9,
            },
            DocumentType.STANDARD: {
                "keywords": ["document", "letter", "report", "memo"],
                "structure_indicators": ["paragraph", "heading", "bullet"],
                "confidence": 0.7,
            },
            DocumentType.COMPLEX: {
                "keywords": ["manual", "catalog", "magazine", "brochure"],
                "structure_indicators": ["image", "multi-column", "irregular"],
                "confidence": 0.6,
            },
        }

    async def classify(self, file_path: str) -> Dict[str, Any]:
        """Classify document type"""
        try:
            # Basic file analysis
            file_info = self._analyze_file(file_path)

            # Extract text for keyword analysis
            text = self._extract_text_sample(file_path)

            # Calculate scores for each type
            scores = {}
            for doc_type, patterns in self.type_patterns.items():
                scores[doc_type] = self._calculate_score(text, patterns, file_info)

            # Determine best match
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]

            # Use classification only if confident enough
            if confidence < 0.5:
                best_type = DocumentType.UNSTRUCTURED
                confidence = 0.3  # Low confidence for unstructured

            return {
                "document_type": best_type,
                "confidence": confidence,
                "scores": scores,
                "file_info": file_info,
            }

        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {
                "document_type": DocumentType.UNSTRUCTURED,
                "confidence": 0.1,
                "scores": {},
                "file_info": {},
                "error": str(e),
            }

    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze file characteristics"""
        from pathlib import Path

        path = Path(file_path)
        stat = path.stat()

        return {
            "size": stat.st_size,
            "extension": path.suffix.lower(),
            "pages": self._count_pages(file_path)
            if path.suffix.lower() == ".pdf"
            else 1,
        }

    def _count_pages(self, pdf_path: str) -> int:
        """Count pages in PDF"""
        try:
            import PyPDF2

            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except:
            return 1  # Default to 1 page on error

    def _extract_text_sample(self, file_path: str) -> str:
        """Extract text sample from file"""
        try:
            from ..common.utils import perform_ocr

            result = perform_ocr(file_path, lang="eng")
            return result["text"].lower()
        except:
            return ""

    def _calculate_score(
        self, text: str, patterns: Dict[str, Any], file_info: Dict[str, Any]
    ) -> float:
        """Calculate classification score"""
        score = 0.0

        # Keyword matching
        for keyword in patterns["keywords"]:
            if keyword in text:
                score += 0.3

        # Structure indicators
        for indicator in patterns["structure_indicators"]:
            if indicator in text:
                score += 0.2

        # File characteristics
        if file_info.get("pages", 1) > 10:
            score += 0.1  # Multi-page documents more likely complex

        if file_info.get("extension") == ".pdf":
            score += 0.1  # PDFs tend to be more structured

        # Normalize score
        return min(score, patterns["confidence"])
