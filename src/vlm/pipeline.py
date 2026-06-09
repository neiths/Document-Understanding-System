import time
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from PIL import Image

from ..common.models import ExtractionRequest, ExtractionResponse, VLMResponse
from ..common.ml_client import MLflowClient, MinIOClient
from ..common.utils import measure_time
from .vlm_client import VLMClient

logger = logging.getLogger(__name__)


class VLMPipeline:
    """Vision-Language Model pipeline for document understanding"""

    def __init__(self, mlflow_client: MLflowClient, minio_client: MinIOClient):
        self.mlflow_client = mlflow_client
        self.minio_client = minio_client

        # Initialize VLM client
        self.vlm_client = VLMClient()

        # Document-specific prompts
        self.prompts = {
            "receipt": self._get_receipt_prompt(),
            "form": self._get_form_prompt(),
            "standard": self._get_standard_prompt(),
            "complex": self._get_complex_prompt(),
            "unstructured": self._get_unstructured_prompt(),
        }

    @measure_time
    async def process(self, request: ExtractionRequest) -> ExtractionResponse:
        """Process document through VLM pipeline"""
        try:
            logger.info(f"Starting VLM processing for {request.file_path}")

            # Determine document type
            doc_type = request.document_type or "unstructured"

            # Get appropriate prompt
            prompt = self.prompts.get(doc_type, self._get_unstructured_prompt())

            # Load image
            image = Image.open(request.file_path)

            # Process with VLM
            vlm_response = await self.vlm_client.process_image(
                image=image, prompt=prompt, document_type=doc_type
            )

            # Parse structured response
            structured_data = self._parse_vlm_response(vlm_response)

            # Calculate confidence
            confidence = self._calculate_confidence(vlm_response, structured_data)

            # Build response data
            response_data = {
                "document_type": doc_type,
                "extracted_text": vlm_response.get("extracted_text", ""),
                "structured_data": structured_data,
                "entities": self._extract_entities(vlm_response),
                "metadata": {
                    "model_used": vlm_response.get("model", "qwen2.5-vl"),
                    "tokens_used": vlm_response.get("tokens_used", 0),
                    "processing_time": vlm_response.get("processing_time", 0),
                },
            }

            # Log metrics to MLflow
            metrics = {
                "vlm_pipeline_confidence": confidence,
                "response_length": len(vlm_response.get("extracted_text", "")),
                "structured_fields": len(structured_data),
                "entities_count": len(self._extract_entities(vlm_response)),
            }

            self.mlflow_client.log_model_metrics("vlm_pipeline", metrics, None)

            logger.info(f"VLM processing completed with confidence {confidence:.2f}")

            return ExtractionResponse(
                success=True,
                data=response_data,
                pipeline_used="vlm",
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error in VLM pipeline: {str(e)}")
            return ExtractionResponse(success=False, error=str(e), pipeline_used="vlm")

    def _parse_vlm_response(self, vlm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VLM response into structured format"""
        try:
            # Try to extract JSON from response
            if "json" in str(vlm_response).lower():
                # Find JSON in response
                import re

                json_match = re.search(r"\{[\s\S]*\}", str(vlm_response))
                if json_match:
                    return json.loads(json_match.group())

            # Fallback: parse based on document type patterns
            text = vlm_response.get("extracted_text", "")
            return self._parse_text_response(text)

        except Exception as e:
            logger.error(f"Error parsing VLM response: {str(e)}")
            return {}

    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse raw text response"""
        structured = {}

        # Extract common patterns
        import re

        # Key-value pairs
        kv_pairs = re.findall(r"([^:]+):\s*(.+)", text)
        for key, value in kv_pairs:
            structured[key.strip().lower()] = value.strip()

        # Bullet points
        bullets = re.findall(r"^\s*[-*]\s*(.+)$", text, re.MULTILINE)
        if bullets:
            structured["items"] = bullets

        # Paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        if paragraphs:
            structured["content"] = [p.strip() for p in paragraphs if p.strip()]

        return structured

    def _calculate_confidence(
        self, vlm_response: Dict[str, Any], structured_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for VLM response"""
        base_confidence = 0.5  # Base confidence

        # Boost confidence based on structured data
        if structured_data:
            field_count = len(structured_data)
            base_confidence += min(field_count * 0.1, 0.4)

        # Check for confidence in response
        if "confidence" in vlm_response:
            base_confidence = max(base_confidence, vlm_response["confidence"])

        # Cap at 1.0
        return min(base_confidence, 1.0)

    def _extract_entities(self, vlm_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract named entities from VLM response"""
        entities = []
        text = vlm_response.get("extracted_text", "")

        # Email
        import re

        email_matches = re.findall(
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text
        )
        for email in email_matches:
            entities.append({"type": "email", "text": email, "confidence": 0.9})

        # Phone numbers
        phone_matches = re.findall(r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})", text)
        for phone in phone_matches:
            entities.append({"type": "phone", "text": phone, "confidence": 0.9})

        # Dates
        date_matches = re.findall(r"(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4})", text)
        for date in date_matches:
            entities.append({"type": "date", "text": date, "confidence": 0.8})

        # Numbers
        number_matches = re.findall(r"\$?(\d+\.?\d*)", text)
        for number in number_matches:
            if len(number) < 10:  # Avoid long sequences
                entities.append({"type": "number", "text": number, "confidence": 0.7})

        return entities

    def _get_receipt_prompt(self) -> str:
        """Get structured prompt for receipt processing"""
        return """
        You are a receipt processing expert. Extract all information from this receipt image and return it as JSON.

        Format your response as:
        {
            "store": "Store name",
            "date": "Transaction date",
            "total": "Total amount",
            "items": [
                {"name": "Item name", "quantity": X, "price": X.XX},
                {"name": "Item name", "quantity": X, "price": X.XX}
            ],
            "payment_method": "Payment method",
            "tax": "Tax amount",
            "subtotal": "Subtotal"
        }

        Extract all visible text and organize it into the appropriate fields. If a field is not present, omit it.
        """

    def _get_form_prompt(self) -> str:
        """Get structured prompt for form processing"""
        return """
        You are a form processing expert. Extract all field values from this form image.

        Format your response as:
        {
            "fields": {
                "field_name": "field_value",
                "field_name": "field_value"
            },
            "sections": {
                "section_name": ["field1", "field2"],
                "section_name": ["field1", "field2"]
            }
        }

        Identify all form fields and their corresponding values. Group fields into logical sections if possible.
        """

    def _get_standard_prompt(self) -> str:
        """Get structured prompt for standard document processing"""
        return """
        You are a document processing expert. Extract and summarize the content of this document.

        Format your response as:
        {
            "title": "Document title",
            "author": "Author name",
            "summary": "Brief summary of content",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "sections": {
                "section_name": "Content",
                "section_name": "Content"
            }
        }

        Maintain the original structure and meaning of the document.
        """

    def _get_complex_prompt(self) -> str:
        """Get structured prompt for complex document processing"""
        return """
        You are a complex document processing expert. This document may contain multiple layouts, images, and mixed content.

        Format your response as:
        {
            "title": "Document title",
            "content_type": "Type of document",
            "main_content": "Main text content",
            "captions": ["Caption 1", "Caption 2"],
            "tables": [
                {
                    "headers": ["Header 1", "Header 2"],
                    "rows": [["Row 1 Col 1", "Row 1 Col 2"], ["Row 2 Col 1", "Row 2 Col 2"]]
                }
            ],
            "structure": "Description of document structure"
        }

        Pay special attention to tables, captions, and the relationship between text and images.
        """

    def _get_unstructured_prompt(self) -> str:
        """Get prompt for unstructured documents"""
        return """
        Extract all visible text from this document and organize it as best as possible.

        Format your response as:
        {
            "extracted_text": "All text in the document",
            "structure": "Description of document organization",
            "key_information": ["Key point 1", "Key point 2"],
            "metadata": {
                "page_count": X,
                "language": "Language detected",
                "format": "Document format"
            }
        }

        Include all visible text and make your best attempt to organize it.
        """
