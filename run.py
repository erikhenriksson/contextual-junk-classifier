import os

os.environ["HF_HOME"] = ".hf/hf_home"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for consistency
from safetensors.torch import load_file
import argparse
from datasets import load_dataset, DatasetDict

from sklearn.preprocessing import LabelEncoder


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

import platt_scaler, predict_fineweb


class ImprovedClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_type = config.pooling_type
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout")
            and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, attention_mask=None):
        if self.pooling_type == "cls":
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        elif self.pooling_type == "mean":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(features.size()).float()
            )
            sum_embeddings = torch.sum(features * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            raise ValueError("Unsupported pooling type. Use 'cls' or 'mean'.")

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomClassificationModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        model_kwargs = {
            "trust_remote_code": True,
        }

        if "stella" in config.model_name_or_path:
            model_kwargs["use_memory_efficient_attention"] = False
            model_kwargs["unpad_inputs"] = False
        self.transformer = AutoModel.from_pretrained(
            config.model_name_or_path, **model_kwargs
        )
        self.classifier = ImprovedClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output, attention_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.pooling_type = kwargs.get("pooling_type", "mean")
        config.num_labels = kwargs.get("num_labels", 2)
        config.model_name_or_path = model_name_or_path

        # Instantiate the model
        model = cls(config)
        if kwargs.get("use_finetuned_weights", False):
            # Load weights from the .safetensors file
            model_weights = load_file(f"{model_name_or_path}/model.safetensors")
            model.load_state_dict(model_weights)

        return model


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
def main(args):

    # Load each split into a Dataset
    data_files = {
        "train": f"data/free_train{'_synth' if args.add_synthetic_data else ''}.jsonl",
        "test": "data/free_test.jsonl",
        "dev": "data/free_dev.jsonl",
    }

    # Load the JSONL files as a DatasetDict
    dataset = DatasetDict(
        {
            split_name: load_dataset(
                "json",
                data_files={split_name: file_path},
                split=split_name,
            )
            for split_name, file_path in data_files.items()
        }
    )

    # Initialize and fit LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset["train"]["label"])

    # Function to encode labels in a dataset
    def encode_labels(example):
        example["label"] = label_encoder.transform([example["label"]])[0]
        return example

    # Apply the transformation to each split
    dataset = dataset.map(encode_labels)

    # Get model path
    saved_model_name = args.finetuned_model_path or (
        "free_finetuned_"
        + args.base_model.replace("/", "_")
        + ("_with_synth" if args.add_synthetic_data else "")
    )

    # If training, load the base model; if not, load the saved model
    load_model_name = args.base_model if args.train else saved_model_name

    num_labels = len(label_encoder.classes_)

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["line"], padding="longest", truncation=True, max_length=512
        )

    dataset = dataset.map(tokenize, batched=True)

    # Shuffle the train split
    dataset["train"] = dataset["train"].shuffle(seed=42)

    print("Example of a tokenized input:")
    print(dataset["train"][0])

    # Choose the appropriate model
    if args.embedding_model:
        model = CustomClassificationModel.from_pretrained(
            load_model_name,
            num_labels=num_labels,
            pooling_type=args.pooling_type,
            use_finetuned_weights=not args.train,
        )

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            load_model_name, num_labels=num_labels
        )

    if args.test or args.train:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=saved_model_name,
            learning_rate=args.learning_rate or 0.00001,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            logging_dir=f"{saved_model_name}/logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=5,
            seed=42,
            bf16=True,
            tf32=True,
            group_by_length=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=lambda pred: compute_metrics(pred, label_encoder),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            label_smoothing=0.1,
        )
        if args.train:
            # Train the model
            trainer.train()
            trainer.save_model(saved_model_name)

        # Evaluate the model on the test set
        eval_result = trainer.evaluate(eval_dataset=dataset["test"])
        print(f"Test set evaluation results: {eval_result}")

    if args.platt or args.platt_tune:
        platt_scaler.run(
            saved_model_name,
            model.to("cuda"),
            tokenizer,
            dataset["test"],
            label_encoder,
            args.platt_tune,
        )

    if args.predict_line:
        # predict just one line with the loaded model
        line = args.predict_line
        inputs = tokenizer(line, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        predicted_class_idx = logits.argmax().item()
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        print(f"Predicted class for line '{line}': {predicted_class}")
        exit()

    if args.predict_fineweb:
        predict_fineweb.run(
            saved_model_name, model.to("cuda"), tokenizer, label_encoder, "clean"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--pooling_type", type=str, default="cls")
    parser.add_argument("--add_synthetic_data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--finetuned_model_path", type=str)
    parser.add_argument("--embedding_model", action="store_true")
    parser.add_argument("--platt", action="store_true")
    parser.add_argument("--platt_tune", action="store_true")
    parser.add_argument("--predict_line")
    parser.add_argument("--predict_fineweb", action="store_true")
    args = parser.parse_args()

    main(args)
