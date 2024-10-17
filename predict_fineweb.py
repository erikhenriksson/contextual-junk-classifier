import os

os.environ["HF_HOME"] = ".hf/hf_home"
from datasets import load_dataset, Dataset
import joblib
import torch
import numpy as np
from scipy.special import softmax


def predict(batch, model, tokenizer, platt_scaler, label_encoder, target_class="clean"):
    # Tokenize the batch
    inputs = tokenizer(
        batch["text"], return_tensors="pt", padding=True, truncation=True
    )

    # Move model and inputs to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Run the model to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the target class logits
    target_class_index = label_encoder.transform([target_class])[0]
    positive_logits = logits[:, target_class_index].cpu().numpy()

    # Apply Platt scaling
    scaled_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[:, 1]

    # Add the scaled probabilities to the batch
    batch["scaled_probs"] = [
        round(prob, 4) for prob in scaled_probs
    ]  # Rounding to two decimals
    print(batch)
    exit()
    return batch


def run(model_name, model, tokenizer, label_encoder, target_class="clean"):
    platt_scaler = joblib.load(f"platt_scaler_{model_name}.joblib")

    # Use streaming to load data row by row
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", split="train", name="sample-10BT", streaming=True
    )

    # Set up parameters
    output_dir = "exquisiteweb"
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    shard_size = 10000  # Set a larger shard size for saving
    batch_size = 64  # Smaller batch size for inference processing
    shard = []
    shard_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            shard_idx = int(f.read().strip())

    # Resume from the next shard if checkpoint exists
    current_row = 0
    for example in dataset:
        # If a checkpoint exists, skip processed rows
        if current_row < shard_idx * shard_size:
            current_row += 1
            continue

        # Accumulate rows until the shard size is reached
        shard.append(example)
        current_row += 1

        if len(shard) == shard_size:
            # Convert shard to Dataset
            shard_dataset = Dataset.from_list(shard)
            modified_batches = []

            # Process in smaller batches
            for i in range(0, shard_size, batch_size):
                batch = shard[i : i + batch_size]
                batch_dataset = Dataset.from_list(batch)

                # Apply prediction function on the batch
                modified_batch = batch_dataset.map(
                    lambda b: predict(
                        b, model, tokenizer, platt_scaler, label_encoder, target_class
                    ),
                    batched=True,
                )

                # Collect the modified batch
                modified_batches.append(modified_batch)

            # Concatenate all modified batches into a single shard
            modified_shard = Dataset.concatenate(modified_batches)

            # Save the modified shard to disk
            modified_shard.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

            # Update the checkpoint file
            shard_idx += 1
            with open(checkpoint_file, "w") as f:
                f.write(str(shard_idx))

            # Clear the shard to start the next one
            shard = []

    # Save any remaining rows as the final shard
    if shard:
        shard_dataset = Dataset.from_list(shard)
        modified_batches = []

        # Process in smaller batches for the last shard
        for i in range(0, len(shard), batch_size):
            batch = shard[i : i + batch_size]
            batch_dataset = Dataset.from_list(batch)

            # Apply prediction function on the batch
            modified_batch = batch_dataset.map(
                lambda b: predict(
                    b, model, tokenizer, platt_scaler, label_encoder, target_class
                ),
                batched=True,
            )

            # Collect the modified batch
            modified_batches.append(modified_batch)

        # Concatenate all modified batches into a single shard
        modified_shard = Dataset.concatenate(modified_batches)

        # Save the modified shard to disk
        modified_shard.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

        # Update the checkpoint file for the last shard
        with open(checkpoint_file, "w") as f:
            f.write(str(shard_idx + 1))
