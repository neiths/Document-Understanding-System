import os
import argparse
import requests
import hashlib
from pathlib import Path
import logging
import mlflow
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "layout_detection": {
        "name": "DocLayout-YOLO",
        "versions": {
            "v1.0": {
                "url": "https://github.com/hfutan/LayoutLMv3-DocLayout/releases/download/v1.0/doclayout-yolo-base.zip",
                "checksum": "sha256:abc123...",
                "size": "250MB"
            }
        }
    },
    "text_detection": {
        "name": "DBNet",
        "versions": {
            "v1.0": {
                "url": "https://github.com/open-mmlab/mmocr/releases/download/v0.6.0/dbnet_resnet18_fpnc_1200e_icdar2015.zip",
                "checksum": "sha256:def456...",
                "size": "150MB"
            }
        }
    },
    "text_recognition": {
        "name": "PARSeq",
        "versions": {
            "v1.0": {
                "url": "https://github.com/google-research/parser/releases/download/v1.0/parseq_imagenet.pth",
                "checksum": "sha256:ghi789...",
                "size": "100MB"
            }
        }
    },
    "kie": {
        "name": "LayoutLMv3",
        "versions": {
            "v1.0": {
                "url": "https://github.com/microsoft/unilm/releases/download/v1.0/layoutlmv3-base-pytorch.tar.gz",
                "checksum": "sha256:jkl012...",
                "size": "500MB"
            }
        }
    },
    "vlm": {
        "name": "Qwen2.5-VL",
        "versions": {
            "qwen2.5-vl-7b": {
                "url": "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-VL-7B/repo?Revision=master&FilePath=qwen2.5-vl-7b",
                "checksum": "sha256:mno345...",
                "size": "14GB",
                "download_method": "modelscope"
            }
        }
    }
}

class ModelDownloader:
    def __init__(self, mlflow_uri=None, minio_config=None):
        self.mlflow_client = None
        self.minio_client = None

        if mlflow_uri:
            self.mlflow_client = mlflow.MlflowClient(mlflow_uri)

        if minio_config:
            self.minio_client = boto3.client(
                's3',
                endpoint_url=minio_config['endpoint'],
                aws_access_key_id=minio_config['access_key'],
                aws_secret_access_key=minio_config['secret_key']
            )

    def download_model(self, model_type, model_version="latest"):
        """Download a model"""
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")

        config = MODEL_CONFIGS[model_type]

        if model_version == "latest":
            model_version = list(config["versions"].keys())[0]

        if model_version not in config["versions"]:
            raise ValueError(f"Version {model_version} not found for model {model_type}")

        model_info = config["versions"][model_version]
        download_dir = Path(f"models/{model_type}")
        download_dir.mkdir(parents=True, exist_ok=True)

        file_path = download_dir / f"{model_type}_{model_version}.zip"

        if file_path.exists():
            logger.info(f"Model already exists at {file_path}")
            return str(file_path)

        logger.info(f"Downloading {model_type} v{model_version}...")

        if model_info.get("download_method") == "modelscope":
            self._download_modelscope_model(model_info, file_path)
        else:
            self._download_http_model(model_info, file_path)

        # Verify checksum
        if "checksum" in model_info:
            self._verify_checksum(file_path, model_info["checksum"])

        logger.info(f"Model downloaded to {file_path}")
        return str(file_path)

    def _download_http_model(self, model_info, file_path):
        """Download model via HTTP"""
        url = model_info["url"]

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rDownloading: {progress:.1f}%", end='')

        print()  # New line after download

    def _download_modelscope_model(self, model_info, file_path):
        """Download model from ModelScope"""
        import modelscope
        from modelscope import snapshot_download

        model_id = "qwen/Qwen2.5-VL-7B"  # Example model ID

        snapshot_download(
            model_id,
            cache_dir="models/cache",
            local_dir=file_path.parent
        )

    def _verify_checksum(self, file_path, expected_checksum):
        """Verify file checksum"""
        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        actual_checksum = f"sha256:{hash_sha256.hexdigest()}"

        if actual_checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch for {file_path}")

    def upload_to_minio(self, model_path, model_type, model_version):
        """Upload model to MinIO"""
        if not self.minio_client:
            raise ValueError("MinIO client not configured")

        bucket_name = "model-registry"
        s3_key = f"models/{model_type}/{model_version}/"

        # Upload files
        if os.path.isfile(model_path):
            self.minio_client.upload_file(
                model_path,
                bucket_name,
                f"{s3_key}{os.path.basename(model_path)}"
            )
        else:
            # Upload directory
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, model_path)
                    s3_path = os.path.join(s3_key, relative_path).replace("\\", "/")

                    self.minio_client.upload_file(
                        local_path,
                        bucket_name,
                        s3_path
                    )

        logger.info(f"Model uploaded to MinIO: s3://{bucket_name}/{s3_key}")

def main():
    parser = argparse.ArgumentParser(description="Download document understanding models")
    parser.add_argument("model_type", choices=list(MODEL_CONFIGS.keys()), help="Type of model to download")
    parser.add_argument("--version", default="latest", help="Model version")
    parser.add_argument("--upload", action="store_true", help="Upload to MinIO after download")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    parser.add_argument("--minio-endpoint", help="MinIO endpoint")
    parser.add_argument("--minio-access-key", help="MinIO access key")
    parser.add_argument("--minio-secret-key", help="MinIO secret key")

    args = parser.parse_args()

    # Initialize MinIO config
    minio_config = None
    if args.upload:
        if not all([args.minio_endpoint, args.minio_access_key, args.minio_secret_key]):
            parser.error("--upload requires --minio-endpoint, --minio-access-key, and --minio-secret-key")

        minio_config = {
            "endpoint": args.minio_endpoint,
            "access_key": args.minio_access_key,
            "secret_key": args.minio_secret_key
        }

    # Initialize downloader
    downloader = ModelDownloader(args.mlflow_uri, minio_config)

    # Download model
    model_path = downloader.download_model(args.model_type, args.version)

    # Upload to MinIO if requested
    if args.upload:
        downloader.upload_to_minio(model_path, args.model_type, args.version)
        print(f"Model uploaded to MinIO and ready for deployment")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()