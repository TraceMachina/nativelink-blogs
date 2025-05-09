"""Demo training code."""

import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import (
    set_seed as transformers_set_seed,
)

warnings.filterwarnings(
    "ignore",
    message = "'pin_memory'",
    category=UserWarning,
)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the model training process.

    Contains all parameters needed to configure the model training,
    including model selection, dataset options, and training hyperparameters.
    """

    model_name: str = "prajjwal1/bert-tiny"
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    output_dir: str | Path | None = None
    batch_size: int = 16
    learning_rate: float = 5e-5
    epochs: int = 2
    num_testing_samples: int = 1000
    seed: int = 42


def set_seeds(seed: int) -> None:
    """Set seeds used during training.

    Args:
        seed: The seed value to use for random number generators
    """
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("All random seeds set to %d", seed)


def _section(title: str) -> None:
    """Helper to print a fancy section title."""
    logger.info("\n%s\n%s\n%s\n", "=" * 80, title, "=" * 80)


def create_tokenize_function(tokenizer: PreTrainedTokenizer) -> callable:
    """Create a tokenization function for the given tokenizer.

    Args:
        tokenizer: The tokenizer to use

    Returns:
        A function that tokenizes examples
    """

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors=None,
        )

    return tokenize_function


def print_model_performance(
    title: str,
    trainer: Trainer,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
) -> dict:
    """Print performance metrics for a binary classification model.

    Args:
        title: Section title for the performance report
        trainer: HuggingFace Trainer with the model
        tokenizer: Tokenizer corresponding to the model
        test_dataset: Dataset to evaluate

    Returns:
        Dictionary containing the calculated metrics
    """
    _section(title)
    logger.info("Evaluating model on test set (size: %d)", len(test_dataset))

    tokenized_test = test_dataset.map(
        create_tokenize_function(tokenizer),
        batched=True,
        desc="Tokenizing test data",
    )

    # Predictions
    predictions = trainer.predict(tokenized_test)
    logits = predictions.predictions
    predicted_classes = np.argmax(logits, axis=1)
    labels = predictions.label_ids

    # Confusion matrix metrics
    true_positives = np.sum((predicted_classes == 1) & (labels == 1))
    true_negatives = np.sum((predicted_classes == 0) & (labels == 0))
    false_positives = np.sum((predicted_classes == 1) & (labels == 0))
    false_negatives = np.sum((predicted_classes == 0) & (labels == 1))

    # Derived metrics
    accuracy = np.mean(predicted_classes == labels)
    loss = predictions.metrics["test_loss"]
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_positives": int(true_positives),
            "true_negatives": int(true_negatives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
        },
    }

    logger.info(
        """
Test accuracy: %.4f (%.2f%%)
Test loss: %.4f

Confusion Matrix:
True Positives: %d
True Negatives: %d
False Positives: %d
False Negatives: %d

Classification Metrics:
Precision: %.4f
Recall: %.4f
F1 Score: %.4f""",
        accuracy,
        accuracy * 100,
        loss,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
    )

    return metrics


def train_classifier(config: TrainingConfig | None = None) -> Path:
    """Train a sequence classification model using HuggingFace Transformers.

    This function loads a pre-trained model and fine-tunes it on a specified
    dataset for sequence classification tasks.

    Args:
        config: Configuration for the training process

    Returns:
        Path: Path to the saved model directory
    """
    _section("Setup")

    config = config or TrainingConfig()

    set_seeds(config.seed)

    output_dir = Path(
        config.output_dir
        or (
            Path("~/.cache/huggingface/hub")
            / config.model_name.replace("/", "_")
        )
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created model directory at: %s", output_dir)

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created checkpoints directory at: %s", checkpoints_dir)

    dataset = load_dataset(config.dataset_name, config.dataset_config)
    logger.info("Finished loading dataset: %s", config.dataset_name)

    total_training_samples = len(dataset["train"])
    logger.info("Total training samples: %d", total_training_samples)

    test_dataset = dataset["train"].select(
        range(
            total_training_samples - config.num_testing_samples,
            total_training_samples,
        )
    )

    logger.info("Using half of dataset for faster training")
    dataset["train"] = dataset["train"].select(
        range(total_training_samples // 2)
    )

    logger.info("Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized_datasets = dataset.map(
        create_tokenize_function(tokenizer),
        batched=True,
    )
    logger.info("Finished tokenizing dataset")

    _section("Setting up trainer")
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
        ),
        args=TrainingArguments(
            eval_strategy="epoch",
            learning_rate=config.learning_rate,
            load_best_model_at_end=True,
            logging_strategy="epoch",
            num_train_epochs=config.epochs,
            output_dir=checkpoints_dir,
            per_device_eval_batch_size=config.batch_size,
            per_device_train_batch_size=config.batch_size,
            save_strategy="epoch",
            weight_decay=0.01,
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    logger.info("Finished setting up trainer instance!")

    print_model_performance(
        "Model Performance Before Fine Tuning", trainer, tokenizer, test_dataset
    )

    _section("Training the model")
    start_time = time.time()
    trainer.train()
    logger.info("Model trained in %.2f seconds", time.time() - start_time)

    _section("Model architecture")
    logger.info(
        """Model type: %s
Total parameters: %s
Trainable parameters: %s""",
        trainer.model.__class__.__name__,
        format(sum(p.numel() for p in trainer.model.parameters()), ","),
        format(
            sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad
            ),
            ",",
        ),
    )

    _section("A sample test inference before saving")
    test_text = "This movie was really good!"
    logger.info("Test text: '%s'", test_text)

    # Retrieve the device of the first parameter tensor in your model.
    # This gives you the exact device where your model currently resides.
    # This makes sure that model fine-tuned remotely is accessible locally if
    # needs be.
    inputs = {
        k: v.to(next(trainer.model.parameters()).device)
        for k, v in tokenizer(test_text, return_tensors="pt").items()
    }

    with torch.no_grad():
        outputs = trainer.model(**inputs)

    logger.info(
        "Predicted class before saving: %d",
        torch.argmax(outputs.logits, dim=1).item(),
    )

    _section("Saving model")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Finished saving model related files!")

    print_model_performance(
        "Performance After Fine Tuning",
        trainer,
        tokenizer,
        test_dataset,
    )

    _section("Done")

    return output_dir


if __name__ == "__main__":
    # Train the model
    out_path = train_classifier()
    logger.info("Model saved to %s", out_path)

    # Load the saved model to verify
    logger.info("Validating saved model...")
    test_text = "This movie was really good!"

    with torch.no_grad():
        outputs = AutoModelForSequenceClassification.from_pretrained(out_path)(
            **AutoTokenizer.from_pretrained(out_path)(
                test_text, return_tensors="pt"
            )
        )

    logger.info(
        "Test inference on '%s' - Predicted class: %d",
        test_text,
        torch.argmax(outputs.logits, dim=1).item(),
    )
