import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os

from ..common.models import ExtractionRequest, ExtractionResponse, OCRPipelineResult
from ..common.ml_client import MLflowClient, MinIOClient
from ..common.utils import measure_time, convert_pdf_to_images
from .layout_detector import LayoutDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .kie_extractor import KIEExtractor

logger = logging.getLogger(__name__)


class OCRPipeline:
    """OCR Pipeline for document understanding"""

    def __init__(self, mlflow_client: MLflowClient, minio_client: MinIOClient):
        self.mlflow_client = mlflow_client
        self.minio_client = minio_client

        # Initialize components
        self.layout_detector = LayoutDetector(mlflow_client)
        self.text_detector = TextDetector(mlflow_client)
        self.text_recognizer = TextRecognizer(mlflow_client)
        self.kie_extractor = KIEExtractor(mlflow_client)

    @measure_time
    async def process(self, request: ExtractionRequest) -> ExtractionResponse:
        """Process document through OCR pipeline"""
        try:
            logger.info(f"Starting OCR processing for {request.file_path}")

            # Convert PDF to images if needed
            if request.file_path.lower().endswith(".pdf"):
                images = convert_pdf_to_images(request.file_path)
                image_paths = [f"temp_page_{i}.jpg" for i, img in enumerate(images)]

                # Save images
                for i, img in enumerate(images):
                    img.save(image_paths[i])
            else:
                image_paths = [request.file_path]

            # Process first image for MVP (can be extended for multi-page)
            image_path = image_paths[0]

            # 1. Layout Detection
            layout_result = await self.layout_detector.detect(image_path)

            # 2. Text Detection
            text_detection_result = await self.text_detector.detect(
                image_path, layout_result.boxes
            )

            # 3. Text Recognition
            text_recognition_results = []
            for bbox in text_detection_result.boxes:
                text_result = await self.text_recognizer.recognize(image_path, bbox)
                text_recognition_results.append(text_result)

            # 4. Key Information Extraction
            kie_result = await self.kie_extractor.extract(
                text_recognition_results, layout_result.document_type
            )

            # Calculate overall confidence
            confidences = [
                layout_result.confidence,
                text_detection_result.confidence,
                sum(r.confidence for r in text_recognition_results)
                / len(text_recognition_results)
                if text_recognition_results
                else 0,
                kie_result.confidence,
            ]
            overall_confidence = sum(confidences) / len(confidences)

            # Build OCR pipeline result
            ocr_result = OCRPipelineResult(
                layout_detection=layout_result,
                text_detection=text_detection_result,
                text_recognition=text_recognition_results,
                kie=kie_result,
                pipeline_confidence=overall_confidence,
                processing_time=0,  # Will be set by measure_time decorator
            )

            # Convert to response format
            response_data = {
                "document_type": layout_result.document_type.value,
                "layout_detection": layout_result.model_dump(),
                "text_blocks": [t.model_dump() for t in text_recognition_results],
                "key_value_pairs": kie_result.key_value_pairs,
                "confidence": overall_confidence,
            }

            # Log metrics to MLflow
            metrics = {
                "ocr_pipeline_confidence": overall_confidence,
                "layout_detection_confidence": layout_result.confidence,
                "text_detection_confidence": text_detection_result.confidence,
                "kie_confidence": kie_result.confidence,
                "text_count": len(text_recognition_results),
            }

            self.mlflow_client.log_model_metrics("ocr_pipeline", metrics, None)

            logger.info(
                f"OCR processing completed with confidence {overall_confidence:.2f}"
            )

            return ExtractionResponse(
                success=True,
                data=response_data,
                pipeline_used="ocr",
                confidence=overall_confidence,
            )

        except Exception as e:
            logger.error(f"Error in OCR pipeline: {str(e)}")
            return ExtractionResponse(success=False, error=str(e), pipeline_used="ocr")

        finally:
            # Clean up temporary files
            if "image_paths" in locals():
                for path in image_paths:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except:
                        pass
