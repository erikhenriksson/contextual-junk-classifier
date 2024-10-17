import os

os.environ["HF_HOME"] = ".hf/hf_home"
from datasets import load_dataset, Dataset
import argparse


def main(args):
    # Use streaming to load data shard by shard
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", split="train", name="sample-10BT", streaming=True
    )

    # Output directory for modified shards
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each shard independently
    for shard_idx, shard in enumerate(dataset):
        # Convert the shard to a Dataset object if needed
        shard_dataset = Dataset.from_dict(shard)

        # Apply your transformation
        modified_shard = shard_dataset.map(
            some_function_that_modifies_data, batched=True
        )

        # Save each modified shard to disk
        modified_shard.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="exquisiteweb")

    args = parser.parse_args()

    main(args)
