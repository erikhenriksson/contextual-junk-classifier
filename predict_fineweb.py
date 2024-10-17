import os
from datasets import load_dataset, Dataset
import joblib
import torch
import warnings
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def predict(
    text_batch, model, tokenizer, platt_scaler, label_encoder, target_class="clean"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store scaled probabilities for each text in the batch
    all_scaled_probs = []

    for text in tqdm(text_batch, desc="Processing text batch"):
        # Split the text into lines
        lines = text.splitlines()

        # Process the lines in smaller batches
        line_probs = []
        for i in range(0, len(lines), 64):
            line_batch = lines[i : i + 64]
            inputs = tokenizer(
                line_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.cpu().numpy()  # Move logits to CPU

            # Extract logits for the target class
            target_class_index = label_encoder.transform([target_class])[0]
            positive_logits = logits[:, target_class_index]

            # Apply Platt scaling on the logits
            scaled_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[
                :, 1
            ]
            line_probs.extend(
                [round(prob, 4) for prob in scaled_probs.tolist()]
            )  # Round to two decimals

            # Free up memory
            del outputs
            torch.cuda.empty_cache()

        # Append the list of line probabilities for this text
        all_scaled_probs.append(line_probs)

    return all_scaled_probs


def run(model_name, model, tokenizer, label_encoder, target_class="clean"):
    platt_scaler = joblib.load(f"platt_scaler_{model_name}.joblib")
    model.half()
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Use streaming to load data row by row
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", name="sample-10BT")

    # Set up parameters
    output_dir = "exquisiteweb"
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    shard_size = 10000  # Set a larger shard size for saving
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
            # Extract the 'text' field and process the lines
            text_batch = [item["text"] for item in shard]
            scaled_probs_batch = predict(
                text_batch, model, tokenizer, platt_scaler, label_encoder, target_class
            )

            # Store the scaled probabilities back in each item of the shard
            for item, scaled_probs in zip(shard, scaled_probs_batch):
                item["line_quality"] = scaled_probs

            # Convert shard to Dataset and save
            shard_dataset = Dataset.from_list(shard)
            shard_dataset.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

            # Update the checkpoint file
            shard_idx += 1
            with open(checkpoint_file, "w") as f:
                f.write(str(shard_idx))

            print(f"Saved shard {shard_idx}")

            # Clear the shard to start the next one
            shard = []

    # Save any remaining rows as the final shard
    if shard:
        text_batch = [item["text"] for item in shard]
        scaled_probs_batch = predict(
            text_batch, model, tokenizer, platt_scaler, label_encoder, target_class
        )

        for item, scaled_probs in zip(shard, scaled_probs_batch):
            item["line_quality"] = scaled_probs

        shard_dataset = Dataset.from_list(shard)
        shard_dataset.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

        with open(checkpoint_file, "w") as f:
            f.write(str(shard_idx + 1))
