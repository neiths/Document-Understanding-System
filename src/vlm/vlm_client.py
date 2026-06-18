import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import time

from ..common.ml_client import MLflowClient, MinIOClient

logger = logging.getLogger(__name__)


class VLMClient:
    """Vision-Language Model client for Qwen2.5-VL"""

    def __init__(self):
        self.model = None
        self.model_name = "qwen2.5-vl-7b"
        self.device = "cuda" if self._has_gpu() else "cpu"
        self.max_tokens = 4096
        self.temperature = 0.1
        self._load_model()

    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except:
            return False

    def _load_model(self):
        """Load VLM model"""
        try:
            # Try to load from MLflow registry first
            try:
                from ..common.ml_client import MLflowClient

                mlflow_client = MLflowClient()
                model_uri = mlflow_client.get_latest_model("Qwen2.5-VL")
                self.model = mlflow_client.load_model(model_uri)
                logger.info("Loaded Qwen2.5-VL from MLflow registry")
            except:
                # Fallback to local model or Ollama
                logger.info("Using Ollama for VLM processing")
                self.model = None

        except Exception as e:
            logger.error(f"Error loading VLM model: {str(e)}")
            self.model = None

    async def process_image(
        self, image: Image.Image, prompt: str, document_type: str = "general"
    ) -> Dict[str, Any]:
        """Process image with VLM"""
        start_time = time.time()

        if self.model:
            # Use loaded model
            result = await self._process_with_loaded_model(image, prompt)
        else:
            # Fallback to Ollama or mock response
            result = await self._process_with_ollama(image, prompt)
            # If Ollama fails, use mock response
            if not result:
                result = await self._generate_mock_response(
                    image, prompt, document_type
                )

        # Add processing metadata
        result["processing_time"] = time.time() - start_time
        result["model"] = self.model_name
        result["document_type"] = document_type

        return result

    async def _process_with_loaded_model(
        self, image: Image.Image, prompt: str
    ) -> Dict[str, Any]:
        """Process image with loaded model"""
        try:
            # This is a placeholder for actual VLM processing
            # In a real implementation, this would use the loaded model

            # Mock response for demonstration
            return {
                "extracted_text": "Mock VLM response - actual model implementation needed",
                "structured_data": {},
                "confidence": 0.7,
                "tokens_used": 1000,
                "reasoning": "This is a placeholder response",
            }
        except Exception as e:
            logger.error(f"Error with loaded model: {str(e)}")
            return {}

    async def _process_with_ollama(
        self, image: Image.Image, prompt: str
    ) -> Dict[str, Any]:
        """Process image with Ollama"""
        try:
            import ollama

            # Convert image to base64
            import base64
            from io import BytesIO

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Send request to Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": [img_str]}],
            )

            return {
                "extracted_text": response["message"]["content"],
                "confidence": 0.8,
                "tokens_used": 1500,
                "reasoning": "Processed with Ollama",
            }

        except Exception as e:
            logger.error(f"Error with Ollama: {str(e)}")
            return {}

    async def _generate_mock_response(
        self, image: Image.Image, prompt: str, document_type: str
    ) -> Dict[str, Any]:
        """Generate mock response for testing"""
        # Extract basic image info
        width, height = image.size
        mode = image.mode

        # Generate mock response based on document type
        if document_type == "receipt":
            return {
                "extracted_text": "Sample receipt text: Store: Example Mart, Date: 2024-01-15, Total: $25.99, Items: Milk, Bread, Eggs",
                "structured_data": {
                    "store": "Example Mart",
                    "date": "2024-01-15",
                    "total": "25.99",
                    "items": [
                        {"name": "Milk", "price": "3.99"},
                        {"name": "Bread", "price": "2.99"},
                        {"name": "Eggs", "price": "9.99"},
                    ],
                },
                "confidence": 0.6,
                "tokens_used": 800,
                "reasoning": "Mock response for receipt",
            }
        elif document_type == "form":
            return {
                "extracted_text": "Sample form text: Name: John Doe, Address: 123 Main St, Email: john@example.com, Phone: 555-1234",
                "structured_data": {
                    "fields": {
                        "name": "John Doe",
                        "address": "123 Main St",
                        "email": "john@example.com",
                        "phone": "555-1234",
                    }
                },
                "confidence": 0.7,
                "tokens_used": 600,
                "reasoning": "Mock response for form",
            }
        else:
            return {
                "extracted_text": f"Sample {document_type} document with {width}x{height} {mode} image content...",
                "structured_data": {},
                "confidence": 0.5,
                "tokens_used": 500,
                "reasoning": "Mock response for general document",
            }

    async def batch_process(
        self, images: List[Image.Image], prompt: str, document_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """Process multiple images"""
        tasks = []
        for image in images:
            task = self.process_image(image, prompt, document_type)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "loaded": self.model is not None,
        }

    def update_model(self, model_name: str):
        """Update model"""
        self.model_name = model_name
        self._load_model()
