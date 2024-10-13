import os

os.environ["HF_HOME"] = ".hf/hf_home"

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from linear_dataset import get_data


# Calculate class weights based on the training dataset
def calculate_class_weights(labels, num_labels):
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_labels), y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)


# Custom model with a classification head
class CustomSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels, use_mean_pooling=True):
        super(CustomSequenceClassification, self).__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.use_mean_pooling = use_mean_pooling
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        if self.use_mean_pooling:
            # Use mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Use the CLS token's embedding
            pooled_output = outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute the weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def __init__(self, *args, label_smoothing, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels from inputs
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute loss with label smoothing
        loss = self.loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Pass label_encoder to use class names in the classification report
def compute_metrics(pred, label_encoder):
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

    # Use label_encoder to get class names
    class_names = label_encoder.classes_

    # Generate the classification report using class names
    class_report = classification_report(labels, preds, target_names=class_names)

    # Print the classification report
    print("Classification Report:\n", class_report)

    # Return metrics
    return {
        "accuracy": accuracy,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
    }


# Main function to run the training process
def run(args):
    # Load and preprocess data
    data, label_encoder = get_data(
        args.multiclass,
        downsample_ratio=args.downsample_clean_ratio,
        add_synthetic_data=args.add_synthetic_data,
    )

    suffix = "_multiclass" if args.multiclass else "_binary"
    class_weights = "_class_weights" if args.use_class_weights else ""
    use_synth = "_synth" if args.add_synthetic_data else ""
    smooth = f"_smoothing-{args.label_smoothing}" if args.label_smoothing > 0.0 else ""
    saved_model_name = f"newdata_{args.base_model.replace('/', '_')}_base_model{suffix}_clean_ratio_{args.downsample_clean_ratio}{class_weights}{smooth}{use_synth}"

    num_labels = len(label_encoder.classes_)

    # Tokenize dataset
    if args.base_model == "jxm/cde-small-v1":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="longest", truncation=True, max_length=512
        )

    dataset = data.map(tokenize, batched=True)

    # Shuffle the train split
    dataset["train"] = dataset["train"].shuffle(seed=42)

    print("Example of a tokenized input:")
    print(dataset["train"][0])

    # Choose the appropriate model
    if args.embedding_model:

        if "stella" in args.base_model:
            base_model = AutoModel.from_pretrained(
                args.base_model,
                trust_remote_code=True,
                use_memory_efficient_attention=False,
                unpad_inputs=False,
            )
        else:
            base_model = AutoModel.from_pretrained(
                "bert-base-uncased",
                trust_remote_code=True,
            )
        model = CustomSequenceClassification(base_model, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model if args.train else saved_model_name, num_labels=num_labels
        )

    # Calculate class weights based on the training labels
    train_labels = np.array(dataset["train"]["label"])
    class_weights = calculate_class_weights(train_labels, num_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=saved_model_name,
        learning_rate=3e-5,  # Adjust this if needed for scaling
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        logging_dir=f"{saved_model_name}/logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        # greater_is_better=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        seed=42,
        fp16=True,
    )

    # Define early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

    if args.use_class_weights:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=lambda pred: compute_metrics(pred, label_encoder),
            callbacks=[early_stopping],
            class_weights=class_weights,
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=lambda pred: compute_metrics(pred, label_encoder),
            callbacks=[early_stopping],
            label_smoothing=args.label_smoothing,
        )

    if args.train:
        trainer.train()

    # Evaluate the model on the test set
    eval_result = trainer.evaluate(eval_dataset=dataset["test"])
    print(f"Test set evaluation results: {eval_result}")

    # Save the best model
    trainer.save_model(saved_model_name)
