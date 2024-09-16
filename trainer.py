import os
import numpy as np
from sklearn.metrics import classification_report, f1_score

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import (
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from model import ContextualXLMRobertaForSequenceClassification
from data import ContextualDataCollator, ContextualTextDataset
from preprocess import get_data


def run(data_path, mode):

    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    train_data, dev_data, test_data, num_labels = get_data(data_path, mode)

    # Create datasets
    train_dataset = ContextualTextDataset(train_data, tokenizer)
    dev_dataset = ContextualTextDataset(dev_data, tokenizer)
    test_dataset = ContextualTextDataset(test_data, tokenizer)

    print(train_dataset[0])

    model = ContextualXLMRobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    data_collator = ContextualDataCollator(tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=1)

        micro_f1 = f1_score(labels, preds, average="micro")
        macro_f1 = f1_score(labels, preds, average="macro")

        class_report = classification_report(labels, preds, output_dict=True)

        print("Classification Report:")
        print(classification_report(labels, preds))

        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "class_report": class_report,
        }

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=5,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        logging_dir="./logs",
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_results)
