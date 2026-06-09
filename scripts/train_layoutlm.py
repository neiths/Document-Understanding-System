import os
import json
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LayoutLMTrainer:
    def __init__(self, model_name="microsoft/layoutlm-base-uncased", num_labels=10):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Move model to device
        self.model.to(self.device)

    def train(
        self,
        train_dataset_path,
        val_dataset_path,
        output_dir="./models/layoutlm_trained",
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        warmup_steps=500,
        logging_steps=100,
        save_steps=500,
    ):
        """Train the LayoutLM model"""

        # Load datasets
        train_dataset = self._load_dataset(train_dataset_path)
        val_dataset = self._load_dataset(val_dataset_path)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)

        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Start MLflow run
        mlflow.start_run(run_name="LayoutLM Training")

        # Log parameters
        mlflow.log_param("model_name", self.model_name)
        mlflow.log_param("num_labels", self.num_labels)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("warmup_steps", warmup_steps)

        # Training loop
        best_f1 = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training
            self.model.train()
            train_loss = 0.0

            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()

                # Log metrics
                if step % logging_steps == 0:
                    avg_loss = train_loss / (step + 1)
                    logger.info(f"Step {step}, Loss: {avg_loss:.4f}")
                    mlflow.log_metric(
                        "train_loss", avg_loss, step=epoch * len(train_loader) + step
                    )

                # Save checkpoint
                if step % save_steps == 0 and step > 0:
                    self._save_checkpoint(
                        output_dir, epoch, step, self.model, optimizer, scheduler
                    )

            # Validation
            avg_val_loss, metrics = self._validate(val_loader)

            # Log validation metrics
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)

            # Save best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")

        # End MLflow run
        mlflow.end_run()

        logger.info(f"Training completed. Best F1 score: {best_f1:.4f}")

        return best_f1

    def _load_dataset(self, dataset_path):
        """Load dataset from JSON file"""
        # This is a simplified version - in practice, you'd use a custom Dataset class
        with open(dataset_path, "r") as f:
            data = json.load(f)

        # Convert to Hugging Face dataset
        from datasets import Dataset

        return Dataset.from_dict(data)

    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                val_loss += loss.item()

                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                predictions.extend(preds.cpu().numpy())
                labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)

        # Flatten predictions and labels
        predictions = [p for pred in predictions for p in pred]
        labels = [l for label in labels for l in label]

        # Calculate metrics (excluding padding token)
        mask = labels != -100
        predictions = np.array(predictions)[mask]
        labels = np.array(labels)[mask]

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return avg_val_loss, metrics

    def _save_checkpoint(self, output_dir, epoch, step, model, optimizer, scheduler):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(
            output_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Log checkpoint to MLflow
        mlflow.log_artifact(checkpoint_path, "checkpoints")


def main():
    parser = argparse.ArgumentParser(
        description="Train LayoutLM model for document understanding"
    )
    parser.add_argument("--train-data", required=True, help="Path to training data")
    parser.add_argument("--val-data", required=True, help="Path to validation data")
    parser.add_argument(
        "--output-dir", default="./models/layoutlm_trained", help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/layoutlm-base-uncased",
        help="Pretrained model name",
    )
    parser.add_argument("--num-labels", type=int, default=10, help="Number of labels")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI"
    )

    args = parser.parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)

    # Initialize trainer
    trainer = LayoutLMTrainer(model_name=args.model_name, num_labels=args.num_labels)

    # Train model
    trainer.train(
        train_dataset_path=args.train_data,
        val_dataset_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
