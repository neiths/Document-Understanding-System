import mlflow
import mlflow.pytorch
import mlflow.transformers
import boto3
from botocore.exceptions import ClientError
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class MLflowClient:
    """MLflow client for model registry operations"""

    def __init__(self, tracking_uri: str, artifact_uri: str = None):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.artifact_uri = artifact_uri or f"{tracking_uri}/artifacts"

        # Initialize S3 client for MinIO
        if artifact_uri and artifact_uri.startswith("http"):
            parsed = self._parse_s3_uri(artifact_uri)
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=parsed["endpoint"],
                aws_access_key_id=parsed["key"],
                aws_secret_access_key=parsed["secret"],
            )
            self.bucket = parsed["bucket"]
        else:
            self.s3_client = None

    def _parse_s3_uri(self, uri: str) -> Dict[str, str]:
        """Parse S3 URI to extract MinIO connection details"""
        # Example: http://minio:9000/your-bucket
        parts = uri.replace("http://", "").replace("https://", "").split("/")
        endpoint = f"http://{parts[0]}"
        bucket = parts[1] if len(parts) > 1 else ""

        return {
            "endpoint": endpoint,
            "bucket": bucket,
            "key": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        }

    def get_latest_model(self, model_name: str, stage: str = "Production") -> str:
        """Get the latest model URI for a given model name"""
        client = mlflow.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")

        # Filter by stage
        production_versions = [v for v in model_versions if v.current_stage == stage]

        if not production_versions:
            raise ValueError(f"No production model found for {model_name}")

        # Get the latest version
        latest_version = max(production_versions, key=lambda x: x.version)
        model_uri = f"models:/{model_name}/{latest_version.version}"

        logger.info(f"Retrieved model {model_name} version {latest_version.version}")
        return model_uri

    def load_model(self, model_uri: str):
        """Load a model from MLflow"""
        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def log_model_metrics(
        self, experiment_name: str, metrics: Dict[str, Any], artifact_path: str = None
    ):
        """Log metrics to MLflow"""
        with mlflow.start_run(experiment_name=experiment_name):
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)


class MinIOClient:
    """MinIO client for model storage operations"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.endpoint = endpoint
        self.bucket = bucket

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=f"http://{endpoint}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Bucket doesn't exist, create it
                self.s3_client.create_bucket(Bucket=self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                raise

    def upload_model(self, local_path: str, model_name: str, version: str) -> str:
        """Upload model artifacts to MinIO"""
        s3_key = f"models/{model_name}/{version}/"

        # Upload directory
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                s3_path = os.path.join(s3_key, relative_path).replace("\\", "/")

                self.s3_client.upload_file(local_file, self.bucket, s3_path)

        logger.info(f"Uploaded model {model_name} v{version} to MinIO")
        return f"s3://{self.bucket}/{s3_key}"

    def download_model(self, model_name: str, version: str, local_path: str):
        """Download model artifacts from MinIO"""
        s3_key = f"models/{model_name}/{version}/"

        # Create local directory
        os.makedirs(local_path, exist_ok=True)

        # List all objects in the prefix
        objects = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=s3_key)

        # Download each file
        for obj in objects.get("Contents", []):
            s3_path = obj["Key"]
            local_file = os.path.join(local_path, s3_path.replace(s3_key, ""))

            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            self.s3_client.download_file(self.bucket, s3_path, local_file)

        logger.info(f"Downloaded model {model_name} v{version} from MinIO")

    def list_models(self) -> Dict[str, list]:
        """List all models in MinIO"""
        models = {}

        # List all model directories
        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix="models/")

        for obj in response.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")

            if len(parts) >= 3:  # models/model-name/version/
                model_name = parts[1]
                version = parts[2]

                if model_name not in models:
                    models[model_name] = []

                models[model_name].append(version)

        return models
