from linear_dataset import get_data
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm import tqdm


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
    if "stella" in args.base_model:
        model = AutoModel.from_pretrained(
            args.local_model,
            trust_remote_code=True,
            use_memory_efficient_attention=False,
            unpad_inputs=False,
        ).to("cuda")

    else:
        model = AutoModel.from_pretrained(args.local_model).to("cuda")
    model.half()
    # Place model in evaluation mode
    model.eval()

    # Determine the index for the "clean" label
    clean_label_index = label_encoder.transform(["clean"])[0]

    # Convert test labels to binary format based on the clean_label_index
    test_labels = np.array(
        [0 if label == clean_label_index else 1 for label in test_dataset["label"]]
    )

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator
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

            # Forward pass
            outputs = model(**inputs)
            logits = (
                outputs.logits if hasattr(outputs, "logits") else outputs[0]
            )  # Access logits directly

            # print the length of each logit
            print(logits.shape)

            # Store logits
            logits_list.append(logits.cpu().numpy())

    # Convert lists to arrays
    logits = np.concatenate(logits_list, axis=0)

    # Use only the logits for the positive class (index 1, assuming the second column corresponds to non-clean class)
    positive_logits = logits[:, 1]

    # Train Platt scaling logistic regression on the logits using the binary test_labels
    platt_scaler = LogisticRegression(C=1e10, solver="lbfgs")
    platt_scaler.fit(positive_logits.reshape(-1, 1), test_labels)

    # Apply the Platt scaler to get calibrated probabilities
    calibrated_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[:, 1]

    # Save texts and calibrated probabilities to a JSON Lines file
    output_file = f"calibrated_results_{args.local_model}.jsonl"
    with open(output_file, "w") as f:
        for text, prob in zip(texts, calibrated_probs):
            json_line = json.dumps({"text": text, "calibrated_probability": prob})
            f.write(json_line + "\n")

    print(f"Calibrated probabilities and texts saved to {output_file}")
    return calibrated_probs
