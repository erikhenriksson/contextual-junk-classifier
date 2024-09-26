import os

os.environ["HF_HOME"] = ".hf/hf_home"

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from dataset import get_data


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    # Calculate confusion matrix (optional)
    conf_matrix = confusion_matrix(labels, preds).tolist()

    # Accuracy
    accuracy = accuracy_score(labels, preds)

    # F1 Score
    f1 = f1_score(labels, preds, average="weighted")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")

    # Precision and Recall
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "f1_macro": f1_macro,
        "f2_micro": f1_micro,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
    }


# Main function to run the training process
def run(args):
    # Load and preprocess data
    data, label_encoder = get_data(
        args.multiclass, downsample_clean=True, downsample_ratio=0.3
    )

    suffix = "_multiclass" if args.multiclass else "_binary"

    num_labels = len(label_encoder.classes_)

    # Tokenize data using XLM-Roberta tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="longest", truncation=True, max_length=512
        )

    dataset = data.map(tokenize, batched=True)

    print("Example of a tokenized input:")
    print(dataset["train"][0])

    # Load XLM-Roberta model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=num_labels
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"base_model{suffix}",
        learning_rate=3e-5,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        logging_dir=f"base_model{suffix}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
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
    trainer.save_model(f"base_model{suffix}")
