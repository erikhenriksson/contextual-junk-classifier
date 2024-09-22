import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch


# Load data from JSON files
def load_data(file_path):
    data_splits = {}
    for split in ["dev", "test", "train"]:
        with open(os.path.join(file_path, f"{split}.json"), "r", encoding="utf-8") as f:
            documents = json.load(f)
            texts = [line for doc in documents for line in doc["text"]]
            labels = [int(label) for doc in documents for label in doc["labels"]]
            data_splits[f"{split}_texts"] = texts
            data_splits[f"{split}_labels"] = labels

    return data_splits


# Function to compute weighted F1 score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}


# Main function to run the training process
def main(data_path, output_dir="base_model"):
    # Load and preprocess data
    data = load_data(data_path)

    # Tokenize data using XLM-Roberta tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    # Create Dataset objects
    train_dataset = Dataset.from_dict(
        {"text": data["train_texts"], "label": data["train_labels"]}
    )
    test_dataset = Dataset.from_dict(
        {"text": data["test_texts"], "label": data["test_labels"]}
    )
    dev_dataset = Dataset.from_dict(
        {"text": data["dev_texts"], "label": data["dev_labels"]}
    )

    # Tokenize the datasets
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    tokenized_dev = dev_dataset.map(tokenize, batched=True)

    # Create DatasetDict
    dataset = DatasetDict(
        {"train": tokenized_train, "test": tokenized_test, "dev": tokenized_dev}
    )

    # Load XLM-Roberta model for sequence classification
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=2
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=42,
    )

    # Define early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test set
    eval_result = trainer.evaluate(eval_dataset=dataset["test"])
    print(f"Test set evaluation results: {eval_result}")

    # Save the best model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    data_path = "data/en"  # Replace with your JSON data file path
    main(data_path)
