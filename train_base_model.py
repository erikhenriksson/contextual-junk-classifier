import os

os.environ["HF_HOME"] = ".hf/hf_home"

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from dataset import get_data


# Function to compute weighted F1 score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}


# Main function to run the training process
def run(args):
    # Load and preprocess data
    data, label_encoder = get_data(args.multiclass)

    num_labels = len(label_encoder.classes_)

    # Tokenize data using XLM-Roberta tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset = data.map(tokenize, batched=True)

    print("Example of a tokenized input:")
    print(dataset["train"][0])

    # Load XLM-Roberta model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=num_labels
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="base_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"base_model/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
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
    trainer.save_model("base_model")
