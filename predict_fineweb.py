import os

os.environ["HF_HOME"] = ".hf/hf_home"
from datasets import load_dataset, Dataset
import joblib
import torch
from scipy.special import softmax


def predict(batch, model, tokenizer, platt_scaler, label_encoder, target_class="clean"):
    # Tokenize the input texts in the batch
    inputs = tokenizer(
        batch["text"], return_tensors="pt", padding=True, truncation=True
    )

    # Ensure the model and tokenizer are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Pass inputs through the model to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [num_samples, num_classes]

    # Get the index for the target class
    target_class_index = label_encoder.transform([target_class])[0]

    # Extract the logits for the target class
    positive_logits = (
        logits[:, target_class_index].cpu().numpy()
    )  # Shape: [num_samples]

    # Apply Platt scaling on the positive logits
    scaled_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[
        :, 1
    ]  # Binary probability for target class

    # Add the binary probabilities to the batch
    batch["scaled_probs"] = (
        scaled_probs.tolist()
    )  # Convert to list for compatibility with the dataset

    print(batch)
    exit()

    return batch


def run(model_name, model, tokenizer, label_encoder):
    model.half()
    model.eval()
    platt_scaler = joblib.load(f"platt_scaler_{model_name}.joblib")

    # Use streaming to load data shard by shard
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", split="train", name="sample-10BT", streaming=True
    )

    # Output directory for modified shards
    output_dir = "exquisiteweb"
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    start_shard_idx = 0

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_shard_idx = int(f.read().strip())

    # Process each shard independently, resuming from the last completed one
    for shard_idx, shard in enumerate(dataset):
        # Skip already processed shards
        if shard_idx < start_shard_idx:
            continue

        shard_dataset = Dataset.from_dict(shard)

        # Use batched=True to process the shard in batches
        modified_shard = shard_dataset.map(
            lambda batch: predict(batch, model, tokenizer, platt_scaler, label_encoder),
            batched=True,
        )

        # Save the modified shard
        modified_shard.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

        # Update the checkpoint file
        with open(checkpoint_file, "w") as f:
            f.write(str(shard_idx + 1))
