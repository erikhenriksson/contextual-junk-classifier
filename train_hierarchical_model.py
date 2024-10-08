import os

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import AutoTokenizer, AutoModel, AutoConfig, PretrainedConfig

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from torch.optim import AdamW

from contextual_dataset import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a combined model
class DocumentClassifier(nn.Module):
    def __init__(
        self,
        class_names,
        base_model,
        freeze_base_model=True,
    ):
        super(DocumentClassifier, self).__init__()
        self.num_labels = len(class_names)
        self.class_names = class_names
        self.line_model = AutoModel.from_pretrained(base_model)
        self.base_model_name = self.line_model.config._name_or_path
        # Optionally freeze the base model
        if freeze_base_model:
            for param in self.line_model.parameters():
                param.requires_grad = False

        # Transformer encoder with multiple layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Layer Normalization and Dropout
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=0.1)  # Dropout probability of 0.1

        # Final linear classification layer
        self.linear = nn.Linear(768, self.num_labels)

        # Batch size for tokenization, embedding extraction, and Transformer
        self.batch_size = 16
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    def forward(self, document_lines):
        all_logits = []  # Store logits from all batches

        # Process document lines in batches of size self.batch_size
        for i in range(0, len(document_lines), self.batch_size):
            # Get the current batch of document lines
            batch_lines = document_lines[i : i + self.batch_size]

            # Step 1: Tokenize the current batch of lines
            encoded_inputs = self.tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.line_model.device)

            # Step 2: Extract embeddings from XLM-Roberta for the current batch
            outputs = self.line_model(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
            )
            embeddings = outputs.last_hidden_state[
                :, 0, :
            ]  # [CLS] token embeddings, Shape: [batch_size, 768]

            # Step 3: Add a batch dimension for transformer encoder input
            embeddings = embeddings.unsqueeze(0)  # Shape: [1, batch_size, 768]

            # Step 4: Pass through Transformer Encoder
            encoded_output = self.transformer_encoder(
                embeddings
            )  # Shape: [1, batch_size, 768]

            # Step 5: Apply Layer Normalization and Dropout
            encoded_output = self.layer_norm(encoded_output)
            encoded_output = self.dropout(encoded_output)

            # Step 6: Apply the final linear classification layer
            logits = self.linear(encoded_output)  # Shape: [1, batch_size, num_labels]

            # Remove the batch dimension and store logits for this batch
            all_logits.append(logits.squeeze(0))  # Shape: [batch_size, num_labels]

        # Step 7: Concatenate logits for the entire document
        all_logits = torch.cat(
            all_logits, dim=0
        )  # Shape: [total_num_lines, num_labels]

        return all_logits

    def save_pretrained(self, save_directory):
        """
        Save the model weights, config, and tokenizer in one .bin file.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save entire model state dict (base model + classifier + custom layers)
        model_save_path = os.path.join(save_directory, "model.bin")
        torch.save(
            {
                "base_model": self.line_model.state_dict(),
                "transformer_encoder": self.transformer_encoder.state_dict(),
                "layer_norm": self.layer_norm.state_dict(),
                "linear": self.linear.state_dict(),
            },
            model_save_path,
        )

        label2id = {label: idx for idx, label in enumerate(self.class_names)}
        id2label = {idx: label for idx, label in enumerate(self.class_names)}

        # Save config with label mappings
        config = PretrainedConfig(
            class_names=self.class_names.tolist(),
            base_model=self.base_model_name,
            label2id=label2id,
            id2label=id2label,
        )
        config.save_pretrained(save_directory)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        print(f"Model, config, and tokenizer saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory, device=None):
        """
        Load the model weights, config, and tokenizer from one .bin file.
        """
        # Load config
        config = AutoConfig.from_pretrained(load_directory)

        # Initialize model using the loaded config
        model = cls(
            class_names=config.class_names,
            base_model=config.base_model,
        )

        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved model state dict from the .bin file
        model_load_path = os.path.join(load_directory, "model.bin")
        model_state = torch.load(model_load_path, map_location=device)

        # Load each part of the model from the state dict
        model.line_model.load_state_dict(model_state["base_model"])
        model.transformer_encoder.load_state_dict(model_state["transformer_encoder"])
        model.layer_norm.load_state_dict(model_state["layer_norm"])
        model.linear.load_state_dict(model_state["linear"])

        # Move the model to the specified device
        model = model.to(device)

        print(f"Model loaded from {load_directory} onto {device}")
        return model


# Function to evaluate model on a given dataset
def evaluate_model(val, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    labels = []
    preds = []

    with torch.no_grad():  # Disable gradient calculation
        # Add a progress bar to the evaluation loop
        with tqdm(total=len(val["texts"]), desc="Evaluating") as pbar:
            for document, label in zip(val["texts"], val["labels"]):
                logits = model(document)
                label = (
                    torch.tensor(label).to(device).unsqueeze(0)
                )  # Shape: [1, num_lines]

                # Calculate loss
                loss = loss_fn(logits.view(-1, model.num_labels), label.view(-1))
                total_loss += loss.item()

                # Get the predicted class indices
                predicted_labels = torch.argmax(logits, dim=-1).view(
                    -1
                )  # Shape: [num_lines]

                # Append true and predicted labels for metrics calculation
                labels.extend(label.view(-1).tolist())
                preds.extend(predicted_labels.tolist())

                # Update progress bar
                pbar.update(1)

    avg_loss = total_loss / len(val["texts"])

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

    # Generate the classification report using class names
    class_report = classification_report(labels, preds, target_names=model.class_names)

    # Print the classification report
    print("Classification Report:\n", class_report)

    # Return metrics
    return avg_loss, {
        "accuracy": accuracy,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
    }


def train_model(
    train,
    val,
    model,
    model_save_path,
    optimizer,
    loss_fn,
    epochs=15,
    patience=5,
    lr_scheduler_ratio=0.95,
    evaluation_steps=500,
):
    # Initialize early stopping parameters
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False
    global_step = 0  # Keep track of the total number of steps

    # Set up a linear learning rate scheduler
    num_training_steps = len(train["texts"]) * epochs  # Total number of steps (batches)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(1 - step / num_training_steps, lr_scheduler_ratio),
    )

    for epoch in range(epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        model.train()  # Set the model to training mode
        total_loss = 0

        # Create a progress bar for the training loop
        with tqdm(
            total=len(train["texts"]), desc=f"Epoch {epoch + 1}/{epochs}"
        ) as pbar:
            for document, label in zip(train["texts"], train["labels"]):
                optimizer.zero_grad()  # Reset gradients

                # Forward pass
                logits = model(document)  # Shape: [1, num_lines, num_labels]

                # Assuming `label` is shape [num_lines] with class indices for each line
                label = (
                    torch.tensor(label).to(device).unsqueeze(0)
                )  # Shape: [1, num_lines]

                # Calculate loss
                loss = loss_fn(
                    logits.view(-1, model.num_labels),
                    label.view(-1),
                )

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update the learning rate scheduler
                scheduler.step()

                total_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"Loss": total_loss / (pbar.n + 1)})
                pbar.update(1)

                global_step += 1  # Increment the global step counter

                # Evaluate and check for early stopping at every `evaluation_steps`
                if global_step % evaluation_steps == 0:
                    val_loss, metrics = evaluate_model(
                        val,
                        model,
                        loss_fn,
                    )
                    print("Dev Loss:", val_loss)
                    print("Dev Metrics:", metrics)

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0

                        model.save_pretrained(
                            model_save_path,
                        )
                        print(f"Best model saved at step {global_step}")
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        early_stop = True
                        print(
                            f"Early stopping triggered at step {global_step} after {patience} evaluations without improvement."
                        )
                        break

                    model.train()  # Switch back to training mode

        # Log epoch-level statistics
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train['texts'])}"
        )

        # If early stopping was triggered inside the loop, break the epoch loop as well
        if early_stop:
            break


def run(args):
    # Load and preprocess data
    data, label_encoder = get_data(args.multiclass, args.downsample_clean_ratio)
    model_save_path = args.base_model + "_hierarchical"
    class_names = label_encoder.classes_
    loss_fn = nn.CrossEntropyLoss()

    if args.train:
        model = DocumentClassifier(
            class_names=class_names,
            base_model=args.base_model,
        ).to(device)

        train_model(
            data["train"],
            data["dev"],
            model,
            model_save_path,
            AdamW(model.parameters(), lr=3e-5, weight_decay=0.01),
            loss_fn,
        )

    model = DocumentClassifier.from_pretrained(model_save_path).to(device)

    print("Evaluating on test set...")

    test_loss, metrics = evaluate_model(
        data["test"],
        model,
        loss_fn,
    )

    print("Test Loss:", test_loss)
    print("Test Metrics:", metrics)
