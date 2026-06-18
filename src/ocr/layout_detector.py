import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import cv2
from PIL import Image
import layoutparser as lp

from ..common.models import LayoutDetectionResult
from ..common.ml_client import MLflowClient

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Layout detection using DocLayout-YOLO"""

    def __init__(self, mlflow_client: MLflowClient):
        self.mlflow_client = mlflow_client
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Load DocLayout-YOLO model"""
        try:
            # Try to get model from MLflow registry
            try:
                model_uri = self.mlflow_client.get_latest_model("DocLayout-YOLO")
                self.model = self.mlflow_client.load_model(model_uri)
            except:
                # Fallback to default model
                logger.info("Using default DocLayout-YOLO model")
                self.model = lp.DocLayoutParser(
                    "hfutan/LayoutLMv3-doclayout-base",
                    layout_model="hfutan/doclayout-yolo",
                )

            self.model.to(self.device)
            logger.info("Layout detector loaded successfully")

        except Exception as e:
            logger.error(f"Error loading layout detector: {str(e)}")
            # Use fallback simple layout detection
            self.model = None

    async def detect(self, image_path: str) -> LayoutDetectionResult:
        """Detect document layout"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            if self.model:
                # Use DocLayout-YOLO
                layout = self.model.detect(image)

                # Extract layout elements
                boxes = []
                document_type = "standard"  # Default

                for element in layout:
                    box_data = {
                        "x": element.coordinates[0],
                        "y": element.coordinates[1],
                        "width": element.coordinates[2] - element.coordinates[0],
                        "height": element.coordinates[3] - element.coordinates[1],
                        "type": element.type.lower(),
                        "score": element.score,
                    }
                    boxes.append(box_data)

                    # Determine document type based on layout
                    if element.type.lower() in ["table", "form"]:
                        document_type = "form"
                    elif element.type.lower() in ["receipt"]:
                        document_type = "receipt"
                    elif element.type.lower() in ["image", "figure"]:
                        document_type = "complex"

                # Calculate confidence based on layout confidence scores
                confidences = [box["score"] for box in boxes]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.5
                )

            else:
                # Fallback: Simple edge-based detection
                boxes = self._simple_layout_detection(image_np)
                document_type = "standard"
                avg_confidence = 0.6  # Moderate confidence for fallback

            return LayoutDetectionResult(
                boxes=boxes, confidence=avg_confidence, document_type=document_type
            )

        except Exception as e:
            logger.error(f"Error in layout detection: {str(e)}")
            # Return minimal layout
            return LayoutDetectionResult(
                boxes=[], confidence=0.1, document_type="unknown"
            )

    def _simple_layout_detection(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """Simple edge-based layout detection as fallback"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # Filter small contours
                    boxes.append(
                        {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "type": "unknown",
                            "score": 0.5,
                        }
                    )

            return boxes
        except:
            return []
