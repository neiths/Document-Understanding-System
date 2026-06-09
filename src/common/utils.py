import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import UploadFile, HTTPException
import pytesseract
from PIL import Image
import pdf2image
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure structured logging with Loguru"""
    from loguru import logger
    logger.remove()  # Remove default handler

    # Console logger
    logger.add(
        lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO"
    )

    # File logger
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO"
    )

    return logger


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to disk"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Save file
        with open(destination, "wb") as buffer:
            buffer.write(upload_file.file.read())

        logger.info(f"Saved file to {destination}")
        return destination
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not save file")


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """Convert PDF pages to images"""
    try:
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt='jpeg',
            thread_count=4
        )
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not convert PDF to images")


def perform_ocr(image_path: str, lang: str = 'eng') -> Dict[str, Any]:
    """Perform OCR on an image"""
    try:
        # Open image
        image = Image.open(image_path)

        # Perform OCR
        text = pytesseract.image_to_string(image, lang=lang)
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

        # Extract bounding boxes and text
        boxes = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Only include confident detections
                box = {
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'text': data['text'][i],
                    'confidence': int(data['conf'][i])
                }
                boxes.append(box)

        return {
            'text': text,
            'boxes': boxes,
            'average_confidence': sum(box['confidence'] for box in boxes) / len(boxes) if boxes else 0
        }
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not perform OCR")


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file metadata"""
    path = Path(file_path)
    stat = path.stat()

    return {
        'name': path.name,
        'size': stat.st_size,
        'extension': path.suffix.lower(),
        'created_time': stat.st_ctime,
        'modified_time': stat.st_mtime
    }


def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")

        # Add execution time to result if it's a dict
        if isinstance(result, dict):
            result['execution_time'] = execution_time

        return result
    return wrapper


def validate_file_type(file_path: str, allowed_extensions: list) -> bool:
    """Validate file extension"""
    ext = Path(file_path).suffix.lower()
    return ext in allowed_extensions


def cleanup_temp_files(file_paths: list):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up {file_path}: {str(e)}")