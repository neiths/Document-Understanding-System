import torch
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import re
from collections import defaultdict

from ..common.models import KIEResult
from ..common.ml_client import MLflowClient

logger = logging.getLogger(__name__)


class KIEExtractor:
    """Key Information Extraction using LayoutLMv3"""

    def __init__(self, mlflow_client: MLflowClient):
        self.mlflow_client = mlflow_client
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

        # Define patterns for different document types
        self.patterns = {
            "receipt": {
                "total": r"(?:total|amount|due)\s*[:\-]?\s*\$?(\d+\.?\d*)",
                "date": r"(?:date|transaction\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4})",
                "items": r"(.+?)\s*\$\s*(\d+\.?\d*)",
                "store": r"(?:store|retailer|vendor)[\s\-:]?\s*(.+)",
                "payment": r"(?:payment|method|card)[\s\-:]?\s*(.+)",
            },
            "form": {
                "name": r"name\s*[:\-]?\s*(.+)",
                "address": r"address\s*[:\-]?\s*(.+)",
                "email": r"email\s*[:\-]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                "phone": r"phone\s*[:\-]?\s*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",
                "id": r"(?:id|identification|license)[\s\-:]?\s*(.+)",
            },
            "standard": {"title": r"^(.+)$", "sections": r"^(.+)$"},
        }

    def _load_model(self):
        """Load LayoutLMv3 model"""
        try:
            # Try to get model from MLflow registry
            try:
                model_uri = self.mlflow_client.get_latest_model("LayoutLMv3")
                self.model = self.mlflow_client.load_model(model_uri)
            except:
                logger.info("Using rule-based KIE")
                self.model = None

            logger.info("KIE extractor loaded successfully")

        except Exception as e:
            logger.error(f"Error loading KIE extractor: {str(e)}")
            self.model = None

    async def extract(self, text_results: List[Any], document_type: str) -> KIEResult:
        """Extract key information from text"""
        try:
            # Combine all text results
            all_text = " ".join([t.text for t in text_results])

            if self.model:
                # Use LayoutLMv3 for KIE
                key_value_pairs = await self._layoutlm3_extract(
                    text_results, document_type
                )
                confidence = 0.8  # High confidence for model-based extraction
            else:
                # Use rule-based extraction
                key_value_pairs = self._rule_based_extract(all_text, document_type)
                confidence = 0.6  # Moderate confidence for rule-based

            # Extract entities
            entities = self._extract_entities(text_results, key_value_pairs)

            return KIEResult(
                key_value_pairs=key_value_pairs,
                confidence=confidence,
                entities=entities,
            )

        except Exception as e:
            logger.error(f"Error in KIE extraction: {str(e)}")
            return KIEResult(key_value_pairs={}, confidence=0, entities=[])

    async def _layoutlm3_extract(
        self, text_results: List[Any], document_type: str
    ) -> Dict[str, Any]:
        """Placeholder for LayoutLMv3-based KIE"""
        # In a real implementation, this would use LayoutLMv3
        # For now, return empty dict
        return {}

    def _rule_based_extract(self, text: str, document_type: str) -> Dict[str, Any]:
        """Rule-based key information extraction"""
        key_value_pairs = {}

        # Get patterns for document type
        doc_patterns = self.patterns.get(document_type, self.patterns["standard"])

        for key, pattern in doc_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

            if matches:
                if key == "items" and len(matches) > 1:
                    # For items, store as list of tuples
                    key_value_pairs[key] = []
                    for i in range(0, len(matches), 2):
                        if i + 1 < len(matches):
                            key_value_pairs[key].append(
                                {
                                    "name": matches[i].strip(),
                                    "price": matches[i + 1].strip(),
                                }
                            )
                else:
                    # For other fields, take first match
                    key_value_pairs[key] = matches[0].strip()

        return key_value_pairs

    def _extract_entities(
        self, text_results: List[Any], key_value_pairs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []

        # Add key-value pairs as entities
        for key, value in key_value_pairs.items():
            if isinstance(value, list):
                for item in value:
                    entities.append(
                        {"type": f"{key}_item", "text": str(item), "confidence": 0.8}
                    )
            else:
                entities.append({"type": key, "text": str(value), "confidence": 0.8})

        # Add additional entities based on text content
        for text_result in text_results:
            text = text_result.text

            # Email
            email_match = re.search(
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text
            )
            if email_match:
                entities.append(
                    {"type": "email", "text": email_match.group(1), "confidence": 0.9}
                )

            # Phone number
            phone_match = re.search(r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})", text)
            if phone_match:
                entities.append(
                    {"type": "phone", "text": phone_match.group(1), "confidence": 0.9}
                )

            # Date
            date_match = re.search(r"(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4})", text)
            if date_match:
                entities.append(
                    {"type": "date", "text": date_match.group(1), "confidence": 0.8}
                )

        return entities

    def _post_process(self, key_value_pairs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extracted key-value pairs"""
        processed = {}

        for key, value in key_value_pairs.items():
            # Clean up whitespace
            if isinstance(value, str):
                value = value.strip()

            # Normalize common formats
            if key == "total" and isinstance(value, str):
                # Remove currency symbols and normalize
                value = re.sub(r"[$,\s]", "", value)
                try:
                    value = float(value)
                except:
                    pass

            processed[key] = value

        return processed
