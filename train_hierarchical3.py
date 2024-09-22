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
    train_texts, train_labels, test_texts, test_labels, dev_texts, dev_labels = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    files = ["dev.json", "test.json", "train.json"]
    for file in files:
        file_path = f"f{file_path}/{file}"
        with open(file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        texts = []
        labels = []

        # Iterate through the documents and process each entry
        for doc in documents:
            lines = doc["text"]
            doc_labels = [(int(x)) for x in doc["labels"]]
            texts.extend(lines)
            labels.extend([int(label) for label in doc_labels])

        if file == "dev.json":
            dev_texts = texts
            dev_labels = labels
        elif file == "test.json":
            test_texts = texts
            test_labels = labels
        else:
            train_texts = texts
            train_labels = labels

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
    train_texts, train_labels, test_texts, test_labels, dev_texts, dev_labels = (
        load_data(data_path)
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
    data_path = "data/en"  # Replace with your JSON data file path
    main(data_path)
