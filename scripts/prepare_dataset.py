import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import shutil
from PIL import Image
import yaml

logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    "sroie": {
        "name": "SROIE (Receipts)",
        "description": "Scanned Receipts Optical Information Extraction",
        "url": "https://github.com/SmilesChang/SROIE2019",
        "download_script": "scripts/download_sroie.py",
        "training_format": "icdar",
        "classes": ["company", "date", "address", "total", "other"]
    },
    "funsd": {
        "name": "FUNSD (Forms)",
        "description": "Form Understanding in Noisy Scanned Documents",
        "url": "https://github.com/DS4SD/ FUNSD",
        "download_script": "scripts/download_funsd.py",
        "training_format": "layoutlm",
        "classes": ["question", "answer", "header", "other"]
    },
    "cord": {
        "name": "CORD (Complex Documents)",
        "description": "CORD (Comprehensive Receipt Dataset)",
        "url": "https://github.com/clovaai/cord",
        "download_script": "scripts/download_cord.py",
        "training_format": "layoutlm",
        "classes": ["company", "date", "address", "items", "total", "payment", "other"]
    }
}

class DatasetPreparer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, dataset_name: str, output_format: str = "layoutlm"):
        """Prepare a dataset for training"""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASET_CONFIGS[dataset_name]
        logger.info(f"Preparing {config['name']} dataset")

        # Download dataset if not present
        dataset_path = self.raw_dir / dataset_name
        if not dataset_path.exists():
            logger.info(f"Downloading {dataset_name}...")
            self._download_dataset(dataset_name, config)

        # Prepare for training
        if output_format == "layoutlm":
            self._prepare_layoutlm_format(dataset_name, dataset_path)
        elif output_format == "icdar":
            self._prepare_icdar_format(dataset_name, dataset_path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        logger.info(f"Dataset {dataset_name} prepared successfully")

    def _download_dataset(self, dataset_name: str, config: Dict[str, Any]):
        """Download dataset using the specified script"""
        script_path = Path(config["download_script"])
        if not script_path.exists():
            raise FileNotFoundError(f"Download script not found: {script_path}")

        # Run download script
        os.system(f"python {script_path}")

    def _prepare_layoutlm_format(self, dataset_name: str, dataset_path: Path):
        """Prepare dataset in LayoutLM format"""
        output_dir = self.processed_dir / dataset_name / "layoutlm"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split into train/val
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Process images and annotations
        images_dir = dataset_path / "images"
        annotations_dir = dataset_path / "annotations"

        # Get all image files
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

        # Split 80/20
        split_index = int(len(image_files) * 0.8)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        # Process training files
        for img_file in train_files:
            self._process_layoutlm_file(img_file, annotations_dir, train_dir)

        # Process validation files
        for img_file in val_files:
            self._process_layoutlm_file(img_file, annotations_dir, val_dir)

        # Create dataset info
        self._create_dataset_info(output_dir, len(train_files), len(val_files))

    def _prepare_icdar_format(self, dataset_name: str, dataset_path: Path):
        """Prepare dataset in ICDAR format (for text detection/recognition)"""
        output_dir = self.processed_dir / dataset_name / "icdar"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        images_dir = dataset_path / "images"
        shutil.copytree(images_dir, output_dir / "images")

        # Convert annotations
        annotations_dir = dataset_path / "annotations"
        self._convert_to_icdar_format(annotations_dir, output_dir / "annotations")

    def _process_layoutlm_file(self, img_file: Path, annotations_dir: Path, output_dir: Path):
        """Process a single file for LayoutLM format"""
        # Load image
        image = Image.open(img_file)
        width, height = image.size

        # Find corresponding annotation
        json_file = annotations_dir / f"{img_file.stem}.json"
        if not json_file.exists():
            logger.warning(f"No annotation file found for {img_file}")
            return

        # Load annotation
        with open(json_file, 'r') as f:
            annotation = json.load(f)

        # Create LayoutLM format
        layoutlm_data = {
            "img_path": str(img_file),
            "height": height,
            "width": width,
            "bboxes": [],
            "labels": [],
            "words": []
        }

        # Process each word
        for item in annotation.get("form", []):
            bbox = item["box"]
            label = item.get("label", "O")

            # Normalize bbox [x1, y1, x2, y2]
            norm_bbox = [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height
            ]

            layoutlm_data["bboxes"].append(norm_bbox)
            layoutlm_data["labels"].append(label)
            layoutlm_data["words"].append(item.get("text", ""))

        # Save
        output_file = output_dir / f"{img_file.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(layoutlm_data, f, indent=2)

    def _convert_to_icdar_format(self, annotations_dir: Path, output_dir: Path):
        """Convert annotations to ICDAR format"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each annotation file
        for json_file in annotations_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                annotation = json.load(f)

            # Convert to ICDAR format
                icdar_data = []
            for item in annotation.get("form", []):
                bbox = item["box"]
                text = item.get("text", "")

                icdar_data.append({
                    "text": text,
                    "bbox": bbox
                })

            # Save
            output_file = output_dir / f"{json_file.stem}.txt"
            with open(output_file, 'w') as f:
                for item in icdar_data:
                    f.write(f"{item['text']}\t")
                    f.write(",".join(map(str, item['bbox'])) + "\n")

    def _create_dataset_info(self, output_dir: Path, train_count: int, val_count: int):
        """Create dataset info file"""
        info = {
            "name": output_dir.parent.name,
            "description": f"Prepared {output_dir.parent.name} dataset",
            "train_count": train_count,
            "val_count": val_count,
            "classes": DATASET_CONFIGS[output_dir.parent.name]["classes"],
            "format": "layoutlm"
        }

        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for model training")
    parser.add_argument("dataset", choices=list(DATASET_CONFIGS.keys()), help="Dataset to prepare")
    parser.add_argument("--format", default="layoutlm", choices=["layoutlm", "icdar"], help="Output format")
    parser.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()

    # Initialize preparer
    preparer = DatasetPreparer(args.data_dir)

    # Prepare dataset
    preparer.prepare_dataset(args.dataset, args.format)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()