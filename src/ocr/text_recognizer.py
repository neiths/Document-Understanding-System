import torch
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from PIL import Image
import numpy as np

from ..common.models import TextRecognitionResult
from ..common.ml_client import MLflowClient

logger = logging.getLogger(__name__)


class TextRecognizer:
    """Text recognition using PARSeq"""

    def __init__(self, mlflow_client: MLflowClient):
        self.mlflow_client = mlflow_client
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Load PARSeq model"""
        try:
            # Try to get model from MLflow registry
            try:
                model_uri = self.mlflow_client.get_latest_model("PARSeq")
                self.model = self.mlflow_client.load_model(model_uri)
            except:
                # Fallback to pytesseract
                logger.info("Using pytesseract for text recognition")
                self.model = None

            logger.info("Text recognizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading text recognizer: {str(e)}")
            self.model = None

    async def recognize(
        self, image_path: str, bbox: Dict[str, Any]
    ) -> TextRecognitionResult:
        """Recognize text within bounding box"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Crop to bounding box
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            cropped_image = image.crop((x, y, x + w, y + h))

            if self.model:
                # Use PARSeq for text recognition
                text, confidence = self._parsq_recognize(cropped_image)
            else:
                # Fallback to pytesseract
                text, confidence = self._tesseract_recognize(cropped_image)

            return TextRecognitionResult(text=text, confidence=confidence, bbox=bbox)

        except Exception as e:
            logger.error(f"Error in text recognition: {str(e)}")
            return TextRecognitionResult(text="", confidence=0, bbox=bbox)

    def _parsq_recognize(self, image: Image.Image) -> tuple:
        """Text recognition using PARSeq (placeholder)"""
        # In a real implementation, this would use PARSeq
        # For now, return empty text with low confidence
        return "", 0.1

    def _tesseract_recognize(self, image: Image.Image) -> tuple:
        """Text recognition using pytesseract"""
        import pytesseract

        # Configure pytesseract
        config = "--oem 3 --psm 6"

        # Get text
        text = pytesseract.image_to_string(image, config=config)

        # Get data for confidence
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )

        # Calculate average confidence
        confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return text.strip(), avg_confidence / 100  # Convert to 0-1

    async def recognize_batch(
        self, image_path: str, bboxes: List[Dict[str, Any]]
    ) -> List[TextRecognitionResult]:
        """Recognize text for multiple bounding boxes"""
        results = []

        for bbox in bboxes:
            result = await self.recognize(image_path, bbox)
            results.append(result)

        return results
