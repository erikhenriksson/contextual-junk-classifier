from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import hierarchical_preprocess
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)
import numpy as np
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_class_weights = False
train = "eval"


# Calculate class weights based on the labels
def calculate_class_weights(labels, num_classes, device):
    # Flatten the list of labels and calculate the class frequencies
    flattened_labels = [item for sublist in labels for item in sublist]
    class_counts = np.bincount(flattened_labels, minlength=num_classes)

    # Inverse of class frequencies as weights
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize weights to sum to 1

    return torch.tensor(class_weights, dtype=torch.float).to(device)


class DocumentClassifier(nn.Module):
    def __init__(self, num_labels):
        super(DocumentClassifier, self).__init__()
        self.line_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        # Transformer encoder with multiple layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Layer Normalization and Dropout
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=0.1)  # Dropout probability of 0.1

        # Final linear classification layer
        self.linear = nn.Linear(768, num_labels)

        # Batch size for tokenization, embedding extraction, and Transformer
        self.batch_size = 16
        self.min_batch_size = (
            4  # Minimum batch size for the last batch to be processed alone
        )
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def forward(self, document_lines):
        """
        Forward pass for the document classifier:
        - Tokenize, extract embeddings, pass through transformer and final classifier.
        """
        all_logits = []  # Store logits from all batches

        # Process document lines in batches of size self.batch_size
        num_lines = len(document_lines)

        # Loop over batches of lines
        i = 0
        while i < num_lines:
            # Determine the size of the current batch
            if i + self.batch_size >= num_lines and num_lines - i < self.min_batch_size:
                # If the last batch is too small, concatenate it with the previous batch
                batch_lines = document_lines[i - self.batch_size : num_lines]
                i = num_lines  # Set i to the end since we are processing the last batch
            else:
                # Otherwise, process normally in self.batch_size chunks
                batch_lines = document_lines[i : i + self.batch_size]
                i += self.batch_size

            # Step 1: Tokenize the current batch of lines
            encoded_inputs = self.tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
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


if train == "llm":

    file_path = "../llm-junklabeling/output/fineweb_annotated_gpt4_multi_2.jsonl"

    # Step 1: Dynamically create label mapping based on the data
    label_to_index = hierarchical_preprocess.build_label_mapping(file_path)

    # Step 2: Process the data using the dynamically generated label mapping
    documents, labels = hierarchical_preprocess.process_data(file_path, label_to_index)

    # Get first 1000 documents for faster training
    documents = documents[:1000]
    labels = labels[:1000]

    # Convert labels to binary
    labels = [[1 if l > 0 else 0 for l in label] for label in labels]

    print(f"Number of documents: {len(documents)}")

elif train == "eval":

    file_path = "eval.json"
    documents, labels = hierarchical_preprocess.process_eval_data(file_path)

# num_labels = len(label_to_index)
num_labels = 2

# Split data into train, test, and validation sets (70% train, 20% test, 10% validation)
train_docs, temp_docs, train_labels, temp_labels = train_test_split(
    documents, labels, test_size=0.3, random_state=42
)
val_docs, test_docs, val_labels, test_labels = train_test_split(
    temp_docs, temp_labels, test_size=0.666, random_state=42
)

print(
    f"Train size: {len(train_docs)}, Val size: {len(val_docs)}, Test size: {len(test_docs)}"
)

class_weights = calculate_class_weights(labels, num_labels, device)

# Instantiate the model
model = DocumentClassifier(num_labels)

# Move the model to the appropriate device (GPU if available)

model.to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss(
    weight=class_weights if use_class_weights else None
)  # Cross-entropy loss for multi-class classification


# Function to evaluate model on a given dataset
def evaluate_model(documents, labels, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():  # Disable gradient calculation
        for document, label in zip(documents, labels):
            logits = model(document)
            label = torch.tensor(label).to(device).unsqueeze(0)  # Shape: [1, num_lines]

            # Calculate loss
            loss = loss_fn(logits.view(-1, num_labels), label.view(-1))
            total_loss += loss.item()

            # Get the predicted class indices
            predicted_labels = torch.argmax(logits, dim=-1).view(
                -1
            )  # Shape: [num_lines]

            # Append true and predicted labels for metrics calculation
            y_true.extend(label.view(-1).tolist())
            y_pred.extend(predicted_labels.tolist())

    avg_loss = total_loss / len(documents)

    # Calculate accuracy, precision, recall, F1 score, and other metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # Generate classification report
    class_report = classification_report(y_true, y_pred)

    print("\nClassification Report:\n", class_report)
    print(f"Accuracy: {accuracy}")
    print(f"Precision (Weighted): {precision}")
    print(f"Recall (Weighted): {recall}")
    print(f"F1 Score (Weighted): {f1_weighted}")

    return avg_loss, accuracy, precision, recall, f1_weighted


from torch.optim.lr_scheduler import LambdaLR


def train_model(
    train_docs,
    train_labels,
    val_docs,
    val_labels,
    model,
    optimizer,
    loss_fn,
    epochs=15,
    patience=5,
    lr_scheduler_ratio=0.95,
    evaluation_steps=100,
):
    # Initialize early stopping parameters
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False
    global_step = 0  # Keep track of the total number of steps

    # Set up a linear learning rate scheduler
    num_training_steps = len(train_docs) * epochs  # Total number of steps (batches)
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
        with tqdm(total=len(train_docs), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for document, label in zip(train_docs, train_labels):
                optimizer.zero_grad()  # Reset gradients

                # Forward pass
                logits = model(document)  # Shape: [1, num_lines, num_labels]

                # Assuming `label` is shape [num_lines] with class indices for each line
                label = (
                    torch.tensor(label).to(device).unsqueeze(0)
                )  # Shape: [1, num_lines]

                # Calculate loss
                loss = loss_fn(logits.view(-1, num_labels), label.view(-1))

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
                    model.eval()  # Switch to evaluation mode
                    val_loss, val_accuracy, _, _, _ = evaluate_model(
                        val_docs, val_labels, model, loss_fn
                    )
                    print(
                        f"Step {global_step}: Val Loss = {val_loss}, Val Accuracy = {val_accuracy}"
                    )

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
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
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_docs)}")

        # If early stopping was triggered inside the loop, break the epoch loop as well
        if early_stop:
            break


# Train the model and evaluate on validation set
train_model(train_docs, train_labels, val_docs, val_labels, model, optimizer, loss_fn)

# Evaluate the model on the test set
test_loss, test_accuracy = evaluate_model(test_docs, test_labels, model, loss_fn)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
