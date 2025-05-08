import torch
import numpy as np
import os
import time
import logging

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)


def print_model_performance(model, trainer, tokenizer, test_dataset):
    """
    Function to print model performance metrics

    Note:
    This inferencing is set up for output shape returned by prajjwal1/bert-tiny
    since that's the default model trained by train_classifier function.
    If you use a different model, you may need to adjust the output shape
    """
    logger.info("\nEvaluating model accuracy on test set...")
    # Load the test set
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=128,  # Set an explicit max length
            truncation=True,
            return_tensors=None,  # Important: don't return tensors yet
        )

    # Tokenize the test dataset
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Run prediction on the test set
    predictions = trainer.predict(tokenized_test)

    # Calculate accuracy using NumPy
    logits = predictions.predictions
    predicted_classes = np.argmax(logits, axis=1)
    labels = predictions.label_ids

    accuracy = np.mean(predicted_classes == labels)

    # Calculate basic metrics
    true_positives = np.sum((predicted_classes == 1) & (labels == 1))
    true_negatives = np.sum((predicted_classes == 0) & (labels == 0))
    false_positives = np.sum((predicted_classes == 1) & (labels == 0))
    false_negatives = np.sum((predicted_classes == 0) & (labels == 1))

    # Precision and recall
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

    logger.info(
        f"""
Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Test loss: {predictions.metrics['test_loss']:.4f}

Confusion Matrix (calculated with NumPy):
True Positives: {true_positives}
True Negatives: {true_negatives}
False Positives: {false_positives}
False Negatives: {false_negatives}

Precision: {precision:.4f}
Recall: {recall:.4f}
"""
    )

    return


def train_classifier(
    model_name="prajjwal1/bert-tiny",
    dataset_name="glue",
    dataset_config="sst2",
    output_dir=None,
    batch_size=16,
    learning_rate=5e-5,
    epochs=2,
):
    if output_dir is None:
        output_dir = os.path.join(
            "~/.cache/huggingface/hub",
            model_name.replace("/", "_"),
            # Default Hugging Face cache directory
        )

    # Expand user path if needed
    if output_dir and output_dir.startswith("~"):
        output_dir = os.path.expanduser(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created model directory at: {output_dir}")

    # Create checkpoints subdirectory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Created checkpoints directory at: {checkpoints_dir}")

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    logger.info(f"Finished loading dataset: {dataset_name}!")

    total_training_samples = len(dataset["train"])
    logger.info(f"Total training samples: {total_training_samples}")

    # Reserve the last 1000 samples for testing
    num_testing_samples = 1000
    test_dataset = dataset["train"].select(
        range(total_training_samples - num_testing_samples, total_training_samples)
    )
    logger.info("Using half of dataset for faster training!")
    dataset["train"] = dataset["train"].select(range(total_training_samples // 2))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Finished loading tokenizer!")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    logger.info("Finished loading model!")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=128,  # Set an explicit max length
            truncation=True,
            return_tensors=None,  # Important: don't return tensors yet
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    logger.info("Finished tokenizing dataset!")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    logger.info("Finished setting up training arguments!")

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    logger.info("Finished setting up trainer instance!")

    logger.info(
        f"""
{"=" * 80}
Model Performance Before Fine Tuning
{"=" * 80}
"""
    )
    print_model_performance(model, trainer, tokenizer, test_dataset)

    # To monitor time taken for training
    start_time = time.time()

    logger.info(
        f"""
{"=" * 80}
Training The Model
{"=" * 80}
"""
    )
    # Train model
    trainer.train()
    logger.info("\nFinished training the model!")

    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Model trained in {training_time:.2f} seconds")

    # After setting up the trainer
    logger.info(
        f"""
{"=" * 80}
Model architecture
{"=" * 80}
Model type: {trainer.model.__class__.__name__}
"""
    )

    # Get parameter counts
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}\n")

    # After training and before saving, run an inference test
    test_text = "This movie was really good!"

    logger.info(
        f"""
{"=" * 80}
A sample test inference before saving
{"=" * 80}
Test text: '{test_text}'
"""
    )

    # Get the device the model is currently on
    # retrieves the device of the first parameter tensor in your model.
    # This gives you the exact device where your model currently resides.
    # This makes sure that model fine-tuned remotely is accessible locally if needs be.
    device = next(model.parameters()).device

    # Ensure inputs are on the same device
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {
        k: v.to(device) for k, v in inputs.items()
    }  # moves all your input tensors to that same device.
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    logger.info(f"Predicted class before saving: {predicted_class}")

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Finished saving model related files!\n")

    logger.info(
        f"""
{"=" * 80}
Model Performance After Fine Tuning On NativeLink Cloud For 3 Minutes
{"=" * 80}
"""
    )
    print_model_performance(model, trainer, tokenizer, test_dataset)

    logger.info("*" * 80)
    logger.info("-" * 80)

    return output_dir


if __name__ == "__main__":

    # Detect if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        # Print NVIDIA GPU information
        logger.info(f"NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Train the model
    # cache_dir = "~/.cache/huggingface/hub"
    output_path = train_classifier()
    logger.info(f"Model saved to {output_path}")

    # Load the saved model to verify
    logger.info("Loading saved model to verify...")
    loaded_model = AutoModelForSequenceClassification.from_pretrained(output_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_path)

    # Count parameters
    total_params = sum(p.numel() for p in loaded_model.parameters())
    trainable_params = sum(
        p.numel() for p in loaded_model.parameters() if p.requires_grad
    )

    logger.info(f"Verification - Loaded model parameters: {total_params:,}")
    logger.info(f"Verification - Loaded trainable parameters: {trainable_params:,}")

    # Optional: Try a simple inference to confirm everything works
    test_text = "This movie was really good!"
    inputs = loaded_tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = loaded_model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    logger.info(f"Test inference on '{test_text}' - Predicted class: {predicted_class}")
