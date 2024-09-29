import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from collections import Counter


# Load JSONL data from file
def load_jsonl(filename, label_key, multiclass):
    def convert(line):
        labels = line[
            (
                label_key
                if "llm_junk_annotations_fixed" not in line
                else "llm_junk_annotations_fixed"
            )
        ]
        if not multiclass:
            labels = [x if x == "clean" else "junk" for x in labels]

        line[label_key] = labels
        return line

    with open(filename, "r") as f:
        return [convert(json.loads(line)) for line in f]


# Encode labels using LabelEncoder
def encode_labels(data, label_key, label_encoder=None):

    all_labels = [label for doc in data for label in doc[label_key]]
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(all_labels)
    for doc in data:
        doc[label_key] = label_encoder.transform(doc[label_key]).tolist()
    return data, label_encoder


# Convert list of dicts to Hugging Face Dataset
def create_hf_dataset(doc_data, label_key):
    data = []
    # Split the "text" key by newlines and associate with same index of "label_key" list, and add to the data
    for doc in doc_data:
        for i, line in enumerate(doc["text"].split("\n")):
            data.append({"text": line, "label": doc[label_key][i]})
    return Dataset.from_dict(
        {key: [doc[key] for doc in data] for key in data[0].keys()}
    )


# Downsample the 'clean' class
def downsample_clean_class(dataset, clean_label_index, downsample_ratio=0.1):
    # Convert the dataset to a list of examples
    dataset_dict = dataset.to_dict()

    # Get all indices for the 'clean' class
    clean_indices = [
        i for i, label in enumerate(dataset_dict["label"]) if label == clean_label_index
    ]

    # Randomly select a subset of 'clean' examples to keep
    np.random.seed(42)  # for reproducibility
    num_clean_to_keep = int(len(clean_indices) * downsample_ratio)
    keep_clean_indices = np.random.choice(
        clean_indices, num_clean_to_keep, replace=False
    )

    # Get the indices for all non-clean labels
    non_clean_indices = [
        i for i in range(len(dataset_dict["label"])) if i not in clean_indices
    ]

    # Combine the indices of non-clean and downsampled clean examples
    final_indices = np.concatenate([non_clean_indices, keep_clean_indices])

    # Create the new downsampled dataset
    downsampled_data = {
        key: [dataset_dict[key][i] for i in final_indices]
        for key in dataset_dict.keys()
    }

    return Dataset.from_dict(downsampled_data)


# Function to print class distribution
def print_class_distribution(dataset, split_name, label_encoder):
    labels = dataset["label"]
    label_counts = Counter(labels)
    print(f"Class distribution for {split_name}:")
    for label_index, count in label_counts.items():
        label_name = label_encoder.inverse_transform([label_index])[0]
        print(f"  {label_name}: {count} examples")
    print()


# Main function to load and preprocess the data
def get_data(multiclass, downsample_clean=False, downsample_ratio=0.1):

    label_key = "llm_junk_annotations"

    # Load and process datasets
    train_data = load_jsonl("data/train.jsonl", label_key, multiclass)
    test_data = load_jsonl("data/test.jsonl", label_key, multiclass)
    dev_data = load_jsonl("data/dev.jsonl", label_key, multiclass)

    # Encode labels and create Hugging Face datasets
    train_data, label_encoder = encode_labels(train_data, label_key)
    test_data, _ = encode_labels(test_data, label_key, label_encoder)
    dev_data, _ = encode_labels(dev_data, label_key, label_encoder)

    # Create DatasetDict with all splits
    dataset_dict = DatasetDict(
        {
            "train": create_hf_dataset(train_data, label_key),
            "test": create_hf_dataset(test_data, label_key),
            "dev": create_hf_dataset(dev_data, label_key),
        }
    )

    # Find the label index for 'clean'
    clean_label_index = label_encoder.transform(["clean"])[0]

    # Downsample the 'clean' class in the train set, if specified
    if downsample_clean:
        dataset_dict["train"] = downsample_clean_class(
            dataset_dict["train"], clean_label_index, downsample_ratio
        )

    # Print class distribution for each split
    print_class_distribution(dataset_dict["train"], "train", label_encoder)
    print_class_distribution(dataset_dict["test"], "test", label_encoder)
    print_class_distribution(dataset_dict["dev"], "dev", label_encoder)

    return dataset_dict, label_encoder
