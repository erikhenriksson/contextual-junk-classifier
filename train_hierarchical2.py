from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import hierarchical_preprocess


# Define a combined model
class DocumentClassifier(nn.Module):
    def __init__(self, num_labels):
        super(DocumentClassifier, self).__init__()
        self.line_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(768, num_labels)
        self.batch_size = 4
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize_lines(self, lines):
        # Tokenize all lines in one go, padding them to the same length
        encoded_inputs = self.tokenizer(
            lines, return_tensors="pt", padding=True, truncation=True
        )
        return encoded_inputs.to(
            self.line_model.device
        )  # Ensure they are moved to the same device

    def extract_line_embeddings(self, encoded_inputs):
        all_embeddings = []
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        # Process in batches
        for i in range(0, input_ids.size(0), self.batch_size):
            batch_input_ids = input_ids[i : i + self.batch_size]
            batch_attention_mask = attention_mask[i : i + self.batch_size]

            outputs = self.line_model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
            pooled_embeddings = outputs.last_hidden_state[
                :, 0, :
            ]  # [CLS] token embeddings
            all_embeddings.append(pooled_embeddings)

        return torch.cat(all_embeddings, dim=0)  # Shape: [num_lines, 768]

    def forward(self, document_lines):
        # Tokenize document lines in one go (handles padding within the batch)
        encoded_inputs = self.tokenize_lines(document_lines)

        # Extract embeddings from XLM-Roberta in batches
        embeddings = self.extract_line_embeddings(encoded_inputs)

        # Add batch dimension for transformer encoder input (1 document at a time)
        embeddings = embeddings.unsqueeze(0)  # Shape: [1, num_lines, 768]

        # Pass through Transformer Encoder
        encoded_output = self.transformer_encoder(
            embeddings
        )  # Shape: [1, num_lines, 768]

        # Apply linear classification layer
        logits = self.linear(encoded_output)  # Shape: [1, num_lines, num_labels]

        # Return logits
        return logits


file_path = "../llm-junklabeling/output/fineweb_annotated_gpt4_multi_2.jsonl"

# Step 1: Dynamically create label mapping based on the data
label_to_index = hierarchical_preprocess.build_label_mapping(file_path)

# Step 2: Process the data using the dynamically generated label mapping
documents, labels = hierarchical_preprocess.process_data(file_path, label_to_index)

# Convert labels to binary
labels = [[1 if l > 0 else 0 for l in label] for label in labels]

print(f"Number of documents: {len(documents)}")

num_labels = len(label_to_index)

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

# Instantiate the model
model = DocumentClassifier(num_labels)

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification


# Function to evaluate model on a given dataset
def evaluate_model(documents, labels, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for document, label in zip(documents, labels):
            logits = model(document)
            label = torch.tensor(label).to(device).unsqueeze(0)  # Shape: [1, num_lines]

            # Calculate loss
            loss = loss_fn(logits.view(-1, num_labels), label.view(-1))
            total_loss += loss.item()

            # Compute accuracy
            predicted_labels = torch.argmax(logits, dim=-1).view(
                -1
            )  # Predicted class indices
            correct += (predicted_labels == label.view(-1)).sum().item()
            total += label.numel()

    avg_loss = total_loss / len(documents)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


# Example training loop with validation
def train_model(
    train_docs, train_labels, val_docs, val_labels, model, optimizer, loss_fn, epochs=5
):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for document, label in zip(train_docs, train_labels):
            optimizer.zero_grad()  # Reset gradients

            print(document.shape)

            # Forward pass
            logits = model(document)  # Shape: [1, num_lines, num_labels]

            # Assuming `label` is shape [num_lines] with class indices for each line
            label = torch.tensor(label).to(device).unsqueeze(0)  # Shape: [1, num_lines]

            # Calculate loss
            loss = loss_fn(logits.view(-1, num_labels), label.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on validation set after each epoch
        val_loss, val_accuracy = evaluate_model(val_docs, val_labels, model, loss_fn)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_docs)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )


# Train the model and evaluate on validation set
train_model(train_docs, train_labels, val_docs, val_labels, model, optimizer, loss_fn)

# Evaluate the model on the test set
test_loss, test_accuracy = evaluate_model(test_docs, test_labels, model, loss_fn)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
