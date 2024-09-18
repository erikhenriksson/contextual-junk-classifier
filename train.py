import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import (
    XLMRobertaForSequenceClassification,
    DebertaV2ForSequenceClassification,
    XLMRobertaTokenizer,
    DebertaV2Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from model_roberta import ContextualXLMRobertaForSequenceClassification
from model_roberta_loss import ContextualLossXLMRobertaForSequenceClassification
from model_deberta import ContextualDebertaV2ForSequenceClassification
from data import ContextualDataCollator, ContextualTextDataset
from preprocess import get_data


def run(args):
    do_train = args.train == "yes"
    model_name = args.model_name
    model_type = args.model_type
    if "roberta" in model_name:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        if model_type == "normal":
            model_cls = XLMRobertaForSequenceClassification
        elif model_type == "contextual-pooling":
            model_cls = ContextualXLMRobertaForSequenceClassification
        elif model_type == "contextual-loss":
            model_cls = ContextualXLMRobertaForSequenceClassification

    elif "deberta" in model_name:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        if model_type == "normal":
            model_cls = DebertaV2ForSequenceClassification
        elif model_type == "contextual-pooling":
            model_cls = ContextualDebertaV2ForSequenceClassification

    llm_train_data, llm_dev_data, llm_test_data, num_labels = get_data(
        args.data_path, "jsonl", args.mode, 0.25, args.line_window
    )

    manual_train_data, manual_dev_data, manual_test_data, _ = get_data(
        "eval.json", "json", args.mode, 1, args.line_window
    )

    if args.data_source == "llm":
        print("Using LLM data to train, manual data to evaluate")
        train_data = llm_train_data
        dev_data = llm_dev_data
        test_data = llm_test_data
        eval_data = manual_train_data + manual_dev_data + manual_test_data
    elif args.data_source == "manual":
        print("Using manual data to train, LLM data to evaluate")
        train_data = manual_train_data
        dev_data = manual_dev_data
        test_data = manual_test_data
        eval_data = llm_test_data

    print("Example train data:", train_data[0])

    # Create datasets
    train_dataset = ContextualTextDataset(train_data, tokenizer)
    dev_dataset = ContextualTextDataset(dev_data, tokenizer)
    test_dataset = ContextualTextDataset(test_data, tokenizer)
    eval_dataset = ContextualTextDataset(eval_data, tokenizer)

    print("Tokenized:", train_dataset[0])

    if do_train:
        model = model_cls.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = model_cls.from_pretrained(
            f"./results_{args.data_source}/{args.load_checkpoint}",
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
        output_dir=f"./results_{args.data_source}",
        evaluation_strategy="steps",
        eval_steps=250,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=250,  # Ensure save_steps aligns with eval_steps
        logging_dir="./logs",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        trainer.save_model()

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results (self):", test_results)

    # Evaluate on the manual test set
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    print("Test results (other)", eval_results)
