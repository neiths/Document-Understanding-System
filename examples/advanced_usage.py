"""
Advanced usage examples for the Document Understanding System
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDocumentClient:
    """Advanced client for the Document Understanding System"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def extract_document_async(
        self, file_path: str, document_type: str = None
    ) -> Dict:
        """Asynchronously extract document"""
        url = f"{self.base_url}/api/v1/extract"

        with open(file_path, "rb") as f:
            form_data = aiohttp.FormData()
            form_data.add_field("file", f, filename=Path(file_path).name)

            if document_type:
                form_data.add_field("document_type", document_type)

            async with self.session.post(url, data=form_data) as response:
                return await response.json()

    async def batch_extract_async(
        self, file_paths: List[str], document_types: List[str] = None
    ) -> List[Dict]:
        """Process multiple documents concurrently"""
        tasks = []

        for i, file_path in enumerate(file_paths):
            doc_type = (
                document_types[i]
                if document_types and i < len(document_types)
                else None
            )
            task = self.extract_document_async(file_path, doc_type)
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    def extract_with_retry(
        self, file_path: str, max_retries: int = 3, delay: float = 1.0
    ) -> Dict:
        """Extract document with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.extract_document_sync(file_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)

    def extract_document_sync(self, file_path: str) -> Dict:
        """Synchronous document extraction"""
        import requests

        url = f"{self.base_url}/api/v1/extract"
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f)}
            response = requests.post(url, files=files)
        return response.json()

    def analyze_performance(self, file_paths: List[str]) -> Dict:
        """Analyze processing performance"""
        start_time = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.extract_document_sync, path): path
                for path in file_paths
            }

            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(
                        {
                            "file": file_path,
                            "success": result["success"],
                            "pipeline": result["pipeline_used"],
                            "confidence": result["confidence"],
                            "processing_time": result["processing_time"],
                        }
                    )
                except Exception as e:
                    results.append(
                        {"file": file_path, "success": False, "error": str(e)}
                    )

        total_time = time.time() - start_time

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        if successful:
            avg_time = sum(r["processing_time"] for r in successful) / len(successful)
            avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        else:
            avg_time = 0
            avg_confidence = 0

        return {
            "total_files": len(file_paths),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_time": total_time,
            "average_time": avg_time,
            "average_confidence": avg_confidence,
            "throughput": len(successful) / total_time if total_time > 0 else 0,
            "results": results,
        }

    def monitor_queue(self, check_interval: float = 5.0) -> None:
        """Monitor processing queue"""
        import requests

        while True:
            try:
                # This would be implemented if the system had a queue endpoint
                # For now, we'll just check system metrics
                response = requests.get(f"{self.base_url}/metrics")
                print(f"Queue check at {time.strftime('%H:%M:%S')}")
                # Parse metrics if needed
            except Exception as e:
                logger.error(f"Error monitoring queue: {e}")

            time.sleep(check_interval)


async def document_pipeline():
    """Example of advanced document processing pipeline"""
    client = AdvancedDocumentClient()

    # Example documents
    documents = [
        "invoice1.pdf",
        "receipt1.jpg",
        "form1.png",
        "contract.pdf",
        "manual.pdf",
    ]

    # Filter existing documents
    existing_docs = [d for d in documents if Path(d).exists()]

    if not existing_docs:
        print("No sample documents found. Please add documents to run this example.")
        return

    print("Running Advanced Document Processing Pipeline")
    print("=" * 50)

    # Process documents asynchronously
    async with client:
        print("\n1. Processing documents concurrently...")
        start_time = time.time()

        results = await client.batch_extract_async(existing_docs)

        processing_time = time.time() - start_time
        print(f"Processed {len(results)} documents in {processing_time:.2f}s")

        # Analyze results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if not isinstance(r, dict) or not r.get("success")]

        print(f"\nResults:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            print("\nSuccessful extractions:")
            for result in successful[:3]:  # Show first 3
                data = result.get("data", {})
                print(
                    f"  - {result.get('pipeline', 'unknown')} pipeline (confidence: {result.get('confidence', 0):.2f})"
                )
                if "document_type" in data:
                    print(f"    Type: {data['document_type']}")

        if failed:
            print("\nFailed extractions:")
            for error in failed[:3]:  # Show first 3
                print(f"  - {error}")


def performance_test():
    """Example of performance testing"""
    client = AdvancedDocumentClient()

    # Create test files if needed
    test_files = []
    for i in range(5):  # Create 5 test files
        test_file = f"test_{i}.pdf"
        if not Path(test_file).exists():
            # Create a dummy PDF for testing
            create_test_pdf(test_file)
        test_files.append(test_file)

    print("\nRunning Performance Test")
    print("=" * 30)

    # Run performance analysis
    stats = client.analyze_performance(test_files)

    print(f"\nPerformance Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Average time per file: {stats['average_time']:.2f}s")
    print(f"  Throughput: {stats['throughput']:.2f} files/sec")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")

    # Clean up test files
    for f in test_files:
        try:
            Path(f).unlink()
        except:
            pass


def create_test_pdf(filename):
    """Create a test PDF file"""
    try:
        from PyPDF2 import PdfWriter
        import io

        writer = PdfWriter()
        writer.add_blank_page(612, 792)

        with open(filename, "wb") as f:
            writer.write(f)
        return True
    except ImportError:
        print("PyPDF2 not available, skipping PDF creation")
        return False


def webhook_integration_example():
    """Example of webhook integration"""
    print("\nWebhook Integration Example")
    print("=" * 30)

    # Example webhook handler
    async def webhook_handler(data: Dict):
        """Handle webhook notification"""
        logger.info(f"Webhook received: {json.dumps(data, indent=2)}")

        # Process the result
        if data.get("success"):
            # Save to database
            save_to_database(data)

            # Send notification
            send_notification(data)

            # Trigger downstream processing
            await downstream_processing(data)

    async def save_to_database(data: Dict):
        """Save extraction result to database"""
        logger.info("Saving to database...")
        # Implementation would depend on your database

    async def send_notification(data: Dict):
        """Send notification about completion"""
        logger.info("Sending notification...")
        # Implementation would depend on your notification system

    async def downstream_processing(data: Dict):
        """Process extracted data further"""
        logger.info("Running downstream processing...")
        # Additional processing logic

    # Example usage
    webhook_data = {
        "file": "document.pdf",
        "success": True,
        "pipeline_used": "ocr",
        "confidence": 0.95,
        "data": {
            "extracted_text": "Sample document text...",
            "key_value_pairs": {"name": "John Doe", "amount": "$100.00"},
        },
    }

    # Run the handler
    asyncio.run(webhook_handler(webhook_data))


if __name__ == "__main__":
    # Run advanced examples
    asyncio.run(document_pipeline())

    # Uncomment to run other examples
    # performance_test()
    # webhook_integration_example()
