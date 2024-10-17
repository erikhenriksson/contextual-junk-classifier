from transformers import (
    DataCollatorWithPadding,
)
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from scipy.special import softmax


def run(model, tokenizer, dataset_test, label_encoder, tune=False):

    def tokenize(batch):
        # Ensure padding and truncation are applied directly in the tokenizer
        return tokenizer(
            batch["line"], padding="max_length", truncation=True, max_length=512
        )

    test_dataset = dataset_test.map(tokenize, batched=True)

    # Remove columns that are not tensors
    test_dataset = test_dataset.remove_columns(["text"])

    # Set dataset format to PyTorch tensors
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Define data collator to handle any remaining padding needs
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model.half()
    model.eval()

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=data_collator
    )

    # Store logits and texts
    logits_list = []
    texts = dataset_test["line"]  # Access original texts directly from the dataset

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

    if tune:
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            "C": [1e-5, 1e-3, 0.1, 1, 10, 1e3, 1e5],
            "solver": ["lbfgs", "newton-cg", "sag", "saga"],
        }

        grid_search = GridSearchCV(
            LogisticRegression(), param_grid, cv=5, scoring="neg_log_loss"
        )
        grid_search.fit(positive_logits.reshape(-1, 1), test_labels)

        best_platt_scaler = grid_search.best_estimator_

        print(best_platt_scaler)
        exit()

    # Apply Platt scaling logistic regression on the binary logits
    platt_scaler = LogisticRegression(C=10, solver="sag")
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
