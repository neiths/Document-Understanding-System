# API Reference

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API doesn't require authentication. For production, consider implementing API keys or OAuth2.

## Endpoints

### 1. Extract Document

Extract text and structured information from a document.

**Endpoint:** `POST /extract`

**Headers:**
```
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Document file (PDF, JPG, PNG, TIFF, BMP)
- `document_type` (optional): Specify document type (standard, form, receipt, complex, unstructured). If not provided, the system will auto-detect.

**Request Body:**
```
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="document.pdf"
Content-Type: application/pdf

(file content)
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "document_type": "form",
    "extracted_text": "Extracted text content...",
    "confidence": 0.85,
    "pipeline_used": "ocr",
    "metadata": {
      "file_size": 1048576,
      "file_type": ".pdf"
    },
    "layout_detection": {
      "boxes": [...],
      "confidence": 0.9
    },
    "text_blocks": [
      {
        "text": "Sample text",
        "confidence": 0.8,
        "bbox": {"x": 100, "y": 200, "width": 300, "height": 50}
      }
    ],
    "key_value_pairs": {
      "name": "John Doe",
      "address": "123 Main St"
    }
  },
  "pipeline_used": "ocr",
  "confidence": 0.85,
  "processing_time": 2.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Error Response (400 Bad Request):**
```json
{
  "success": false,
  "error": "Unsupported file type. Please upload PDF, JPG, PNG, TIFF, or BMP."
}
```

**Error Response (500 Internal Server Error):**
```json
{
  "success": false,
  "error": "Model processing failed"
}
```

### 2. Health Check

Check if the API gateway is running.

**Endpoint:** `GET /health`

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": 1705223400.123,
  "version": "1.0.0"
}
```

### 3. List Models

List all available models in the model registry.

**Endpoint:** `GET /models`

**Success Response (200 OK):**
```json
{
  "models": {
    "DocLayout-YOLO": ["v1.0", "v2.0"],
    "DBNet": ["v1.0"],
    "PARSeq": ["v1.0"],
    "LayoutLMv3": ["v1.0", "v2.0"],
    "Qwen2.5-VL": ["qwen2.5-vl-7b"]
  }
}
```

### 4. Get Pipelines

Get information about available processing pipelines.

**Endpoint:** `GET /pipelines`

**Success Response (200 OK):**
```json
{
  "pipelines": {
    "ocr": {
      "status": "active",
      "description": "High confidence OCR pipeline for standard forms"
    },
    "vlm": {
      "status": "active",
      "description": "Vision-Language Model pipeline for complex layouts"
    }
  }
}
```

### 5. Metrics

Prometheus metrics endpoint.

**Endpoint:** `GET /metrics`

**Response:**
Plain text format with Prometheus metrics.

## Document Types

The system supports the following document types:

1. **standard**: Standard documents with clear structure
2. **form**: Forms with labeled fields
3. **receipt**: Receipts with total, date, items, etc.
4. **complex**: Complex layouts with multiple columns, images
5. **unstructured**: Documents without clear structure

## Response Fields

### Extraction Response

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether extraction was successful |
| data | object | Extracted data (see below) |
| pipeline_used | string | Pipeline that processed the document (ocr or vlm) |
| confidence | float | Overall confidence score (0-1) |
| processing_time | float | Time taken in seconds |
| timestamp | string | ISO 8601 timestamp |

### OCR Pipeline Data

| Field | Type | Description |
|-------|------|-------------|
| document_type | string | Type of document detected |
| layout_detection | object | Layout detection results |
| text_blocks | array | Extracted text with positions |
| key_value_pairs | object | Extracted key-value pairs |

### VLM Pipeline Data

| Field | Type | Description |
|-------|------|-------------|
| document_type | string | Type of document detected |
| extracted_text | string | All extracted text |
| structured_data | object | Structured data extracted |
| entities | array | Named entities found |
| metadata | object | Additional metadata |

## Rate Limiting

The API implements rate limiting:

- General requests: 10 requests per second
- File uploads: 2 requests per second

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid file type or parameters |
| 500 | Internal Server Error - Processing failed |
| 413 | Payload Too Large - File too large |

## Example Usage

### cURL

```bash
# Extract from PDF
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@document.pdf"

# Extract from image with specific type
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "file=@receipt.jpg" \
  -F "document_type=receipt"
```

### Python

```python
import requests

# Extract from file
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/extract',
        files={'file': f}
    )
    result = response.json()

if result['success']:
    print(f"Extracted text: {result['data']['extracted_text']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Error: {result['error']}")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/api/v1/extract', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Extracted:', data.data);
    } else {
        console.error('Error:', data.error);
    }
});
```

## Webhook Support

You can configure webhooks to receive notifications when processing is complete:

```json
POST /api/v1/extract
{
  "file": "...",
  "webhook_url": "https://your-domain.com/webhook"
}
```

The system will send a POST request to your webhook URL when processing is complete.

## Batch Processing

For multiple files, use the batch endpoint:

**Endpoint:** `POST /extract/batch`

```json
{
  "files": ["file1.pdf", "file2.jpg"],
  "document_type": "form"
}
```

## SDK Integration

The system provides SDKs for popular languages:

### Python SDK

```python
from document_understanding import DocumentProcessor

processor = DocumentProcessor("http://localhost:8000")
result = processor.extract("document.pdf")
```

### Node.js SDK

```javascript
const { DocumentProcessor } = require('document-understanding-sdk');

const processor = new DocumentProcessor('http://localhost:8000');
const result = await processor.extract('document.pdf');
```