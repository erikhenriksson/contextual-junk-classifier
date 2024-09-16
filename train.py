import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import (
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from model import ContextualXLMRobertaForSequenceClassification
from data import ContextualDataCollator, ContextualTextDataset
from preprocess import get_data
from preprocess_eval import get_eval_data


def run(data_path, mode, do_train):

    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    # train_data, dev_data, test_data, num_labels = get_data(data_path, mode)

    # eval_data = get_eval_data("eval.json", mode)

    _, _, eval_data, num_labels = get_data(data_path, mode)
    train_data, dev_data, test_data, num_labels = get_eval_data("eval.json", mode, True)

    # Create datasets
    train_dataset = ContextualTextDataset(train_data, tokenizer)
    dev_dataset = ContextualTextDataset(dev_data, tokenizer)
    test_dataset = ContextualTextDataset(test_data, tokenizer)
    eval_dataset = ContextualTextDataset(eval_data, tokenizer)

    print(train_dataset[0])
    print(eval_dataset[0])

    if do_train:
        model = ContextualXLMRobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    else:
        model = ContextualXLMRobertaForSequenceClassification.from_pretrained(
            "./results2/checkpoint-2250"
        )

    data_collator = ContextualDataCollator(tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=1)

        # Overall accuracy
        accuracy = accuracy_score(labels, preds)

        # Precision, Recall, F1-score (weighted average)
        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(labels, preds, average="weighted")
        )

        # Precision, Recall, F1-score (macro average)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )

        # Per-class precision, recall, F1-score
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(labels, preds, average=None)
        )

        # Classification report
        class_report = classification_report(
            labels, preds, target_names=["clean", "junk"]
        )

        precision_per_class = precision_per_class.tolist()
        recall_per_class = recall_per_class.tolist()
        f1_per_class = f1_per_class.tolist()
        support_per_class = support_per_class.tolist()

        print("Classification Report:")
        print(class_report)

        return {
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "support_per_class": support_per_class,
        }

    training_args = TrainingArguments(
        output_dir="./results2",
        evaluation_strategy="steps",
        eval_steps=250,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=250,  # Ensure save_steps aligns with eval_steps
        logging_dir="./logs",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
    )

    # Initialize trainer with the EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    if do_train:
        trainer.train()

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_results)

    # Evaluate on the test set
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    print("Manual evaluation results:", eval_results)
