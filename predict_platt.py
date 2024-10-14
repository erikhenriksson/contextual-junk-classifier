from linear_dataset import get_data
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from scipy.special import softmax


def run(args):
    # Load and preprocess data
    data, label_encoder = get_data(args.multiclass, 1, False)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        # Ensure padding and truncation are applied directly in the tokenizer
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=512
        )

    # Tokenize dataset
    dataset_test = data["test"]
    test_dataset = dataset_test.map(tokenize, batched=True)

    # Remove columns that are not tensors
    test_dataset = test_dataset.remove_columns(["text"])

    # Set dataset format to PyTorch tensors
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Define data collator to handle any remaining padding needs
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load model
    # Load the model with the classification head
    if "stella" in args.base_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.local_model,
            trust_remote_code=True,
            use_memory_efficient_attention=False,
            unpad_inputs=False,
        ).to("cuda")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.local_model).to(
            "cuda"
        )

    model.half()
    model.eval()
    print(model)

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=data_collator
    )

    # Store logits and texts
    logits_list = []
    texts = dataset_test["text"]  # Access original texts directly from the dataset

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            # Move batch to device (if using GPU)
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }

            # Forward pass through the fine-tuned model with classification head
            outputs = model(**inputs)
            logits = outputs.logits  # Directly access logits (shape: [batch_size, 9])

            # Store logits
            logits_list.append(logits.cpu().numpy())

    # Convert lists to arrays
    logits = np.concatenate(logits_list, axis=0)  # Shape: [num_samples, 9]

    # Calculate original probabilities using softmax
    original_probs = softmax(logits, axis=1)  # Shape: [num_samples, 9]

    # For binary calculation: Use logits for the specific target class (e.g., "clean")
    clean_label_index = label_encoder.transform(["clean"])[0]
    target_class_index = clean_label_index  # Index of the class for binary separation
    positive_logits = logits[:, target_class_index]  # Shape: [num_samples]

    # Transform labels to binary based on the target class
    test_labels = np.array(
        [1 if label == target_class_index else 0 for label in test_dataset["label"]]
    )

    # Apply Platt scaling logistic regression on the binary logits
    platt_scaler = LogisticRegression(C=1e10, solver="lbfgs")
    platt_scaler.fit(positive_logits.reshape(-1, 1), test_labels)

    # Apply the Platt scaler to get calibrated probabilities
    calibrated_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[:, 1]

    # Save texts, original probabilities, and calibrated probabilities to a JSON Lines file
    output_file = f"calibrated_results_{args.local_model}.jsonl"
    with open(output_file, "w") as f:
        for text, orig_prob, cal_prob in zip(texts, original_probs, calibrated_probs):
            json_line = json.dumps(
                {
                    "text": text,
                    "original_probability": float(
                        orig_prob[target_class_index]
                    ),  # Save only the target class prob
                    "calibrated_probability": float(cal_prob),
                }
            )
            f.write(json_line + "\n")

    print(f"Original and calibrated probabilities saved to {output_file}")
    return calibrated_probs
