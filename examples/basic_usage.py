"""
Basic usage examples for the Document Understanding System
"""

import requests
import json
import time
from pathlib import Path

class DocumentUnderstandingClient:
    """Client for interacting with the Document Understanding System API"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def extract_document(self, file_path, document_type=None):
        """Extract text from a document"""
        url = f"{self.base_url}/api/v1/extract"

        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            data = {}

            if document_type:
                data['document_type'] = document_type

            response = requests.post(url, files=files, data=data)

        return response.json()

    def get_health(self):
        """Check system health"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()

    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/api/v1/models"
        response = requests.get(url)
        return response.json()

    def get_pipelines(self):
        """Get available pipelines"""
        url = f"{self.base_url}/api/v1/pipelines"
        response = requests.get(url)
        return response.json()


def main():
    # Initialize client
    client = DocumentUnderstandingClient()

    print("Document Understanding System - Basic Usage Examples")
    print("=" * 50)

    # 1. Check system health
    print("\n1. Checking system health...")
    try:
        health = client.get_health()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. List available pipelines
    print("\n2. Getting available pipelines...")
    try:
        pipelines = client.get_pipelines()
        for name, info in pipelines['pipelines'].items():
            print(f"- {name}: {info['description']}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. List available models
    print("\n3. Getting available models...")
    try:
        models = client.list_models()
        for model, versions in models['models'].items():
            print(f"- {model}: {', '.join(versions)}")
    except Exception as e:
        print(f"Error: {e}")

    # Example: Process a document
    # Note: You'll need to provide actual document files
    document_path = "sample_document.pdf"  # Change to your document path

    if Path(document_path).exists():
        print(f"\n4. Processing document: {document_path}")

        # Try automatic classification
        try:
            result = client.extract_document(document_path)
            print(f"Pipeline used: {result['pipeline_used']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Processing time: {result['processing_time']:.2f}s")

            if result['success']:
                data = result['data']
                print(f"\nDocument type: {data.get('document_type', 'unknown')}")
                if 'extracted_text' in data:
                    print(f"Extracted text (first 200 chars): {data['extracted_text'][:200]}...")
                if 'key_value_pairs' in data:
                    print("Key-value pairs:")
                    for key, value in data['key_value_pairs'].items():
                        print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error processing document: {e}")
    else:
        print(f"\n4. Document not found: {document_path}")
        print("Skipping document processing example.")


def batch_processing_example():
    """Example of processing multiple documents"""
    print("\n\nBatch Processing Example:")
    print("=" * 30)

    client = DocumentUnderstandingClient()

    # Example document list
    documents = [
        "document1.pdf",
        "document2.jpg",
        "document3.png"
    ]

    results = []

    for doc_path in documents:
        if Path(doc_path).exists():
            print(f"Processing {doc_path}...")
            try:
                result = client.extract_document(doc_path)
                results.append({
                    'file': doc_path,
                    'success': result['success'],
                    'pipeline': result['pipeline_used'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                results.append({
                    'file': doc_path,
                    'success': False,
                    'error': str(e)
                })

    # Summary
    print("\nBatch processing summary:")
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully processed: {successful}/{len(results)}")

    for result in results:
        if result['success']:
            print(f"  {result['file']}: {result['pipeline']} (confidence: {result['confidence']:.2f})")
        else:
            print(f"  {result['file']}: Failed - {result.get('error', 'Unknown error')}")


def webhook_example():
    """Example of setting up webhook processing"""
    print("\n\nWebhook Example:")
    print("=" * 20)

    # This is a conceptual example - actual implementation depends on your use case
    webhook_url = "https://your-server.com/webhook"

    data = {
        "file": "document.pdf",
        "webhook_url": webhook_url,
        "metadata": {
            "user_id": "123",
            "priority": "normal"
        }
    }

    print(f"Setting up webhook at: {webhook_url}")
    print("The system will send a POST request to this URL when processing is complete.")


if __name__ == "__main__":
    main()

    # Uncomment to run additional examples
    # batch_processing_example()
    # webhook_example()