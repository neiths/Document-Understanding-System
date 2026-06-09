import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import cv2
from PIL import Image
from ..common.models import TextDetectionResult
from ..common.ml_client import MLflowClient

logger = logging.getLogger(__name__)


class TextDetector:
    """Text detection using DBNet"""

    def __init__(self, mlflow_client: MLflowClient):
        self.mlflow_client = mlflow_client
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Load DBNet model"""
        try:
            # Try to get model from MLflow registry
            try:
                model_uri = self.mlflow_client.get_latest_model("DBNet")
                self.model = self.mlflow_client.load_model(model_uri)
            except:
                # Fallback to pytesseract
                logger.info("Using pytesseract for text detection")
                self.model = None

            logger.info("Text detector loaded successfully")

        except Exception as e:
            logger.error(f"Error loading text detector: {str(e)}")
            self.model = None

    async def detect(self, image_path: str, layout_boxes: List[Dict[str, Any]] = None) -> TextDetectionResult:
        """Detect text in image"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            if self.model:
                # Use DBNet for text detection
                # This is a placeholder - actual implementation would use DBNet
                text_boxes = self._dbnet_detect(image_np)
            else:
                # Fallback to pytesseract
                text_boxes = self._tesseract_detect(image_np)

            # Filter boxes based on layout if provided
            if layout_boxes:
                text_boxes = self._filter_by_layout(text_boxes, layout_boxes)

            # Calculate confidence
            confidences = [box.get("confidence", 0.8) for box in text_boxes]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return TextDetectionResult(
                text_blocks=text_boxes,
                confidence=avg_confidence
            )

        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            return TextDetectionResult(
                text_blocks=[],
                confidence=0
            )

    def _dbnet_detect(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """Placeholder for DBNet text detection"""
        # In a real implementation, this would use DBNet
        # For now, return empty list
        return []

    def _tesseract_detect(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """Text detection using pytesseract"""
        import pytesseract

        # Configure pytesseract
        config = '--oem 3 --psm 6'

        # Get text data
        data = pytesseract.image_to_data(
            image_np,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        # Extract bounding boxes
        boxes = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Only include confident detections
                box = {
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "width": data['width'][i],
                    "height": data['height'][i],
                    "text": data['text'][i],
                    "confidence": int(data['conf'][i]) / 100  # Convert to 0-1
                }
                boxes.append(box)

        return boxes

    def _filter_by_layout(self, text_boxes: List[Dict[str, Any]], layout_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter text boxes based on layout regions"""
        filtered_boxes = []

        for text_box in text_boxes:
            # Check if text box is within any layout box
            for layout_box in layout_boxes:
                if self._is_inside(text_box, layout_box):
                    filtered_boxes.append(text_box)
                    break

        return filtered_boxes

    def _is_inside(self, box1: Dict[str, Any], box2: Dict[str, Any]) -> bool:
        """Check if box1 is inside box2"""
        x1, y1, w1, h1 = box1["x"], box1["y"], box1["width"], box1["height"]
        x2, y2, w2, h2 = box2["x"], box2["y"], box2["width"], box2["height"]

        # Check if box1 center is inside box2
        center_x1 = x1 + w1 / 2
        center_y1 = y1 + h1 / 2

        return (x2 <= center_x1 <= x2 + w2 and
                y2 <= center_y1 <= y2 + h2)