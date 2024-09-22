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


# Load data from JSON file
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    texts = []
    labels = []

    # Iterate through the documents and process each entry
    for doc in documents:
        lines = doc["text"].split("\n")  # Split text into lines
        doc_labels = doc["labels"].split("\n")  # Split labels into lines
        texts.extend(lines)
        labels.extend(
            [int(label) for label in doc_labels]
        )  # Convert label strings to integers

    return texts, labels


# Function to split the data into train, test, and dev sets
def split_data(texts, labels):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, stratify=labels, random_state=42
    )
    test_texts, dev_texts, test_labels, dev_labels = train_test_split(
        temp_texts, temp_labels, test_size=1 / 3, stratify=temp_labels, random_state=42
    )

    return train_texts, train_labels, test_texts, test_labels, dev_texts, dev_labels


# Function to compute weighted F1 score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}


# Main function to run the training process
def main(data_path, output_dir="base_model"):
    # Load and preprocess data
    texts, labels = load_data(data_path)
    train_texts, train_labels, test_texts, test_labels, dev_texts, dev_labels = (
        split_data(texts, labels)
    )

    # Tokenize data using XLM-Roberta tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    # Create Dataset objects
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    dev_dataset = Dataset.from_dict({"text": dev_texts, "label": dev_labels})

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
    data_path = "eval.json"  # Replace with your JSON data file path
    main(data_path)
