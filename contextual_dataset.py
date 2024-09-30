import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random


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
        for i, label in enumerate(labels):
            if label in ["noise", "other junk", "code"]:
                labels[i] = "other junk"
        if not multiclass:
            labels = [x if x == "clean" else "junk" for x in labels]

        line[label_key] = labels
        return line

    with open(filename, "r") as f:
        return [convert(json.loads(line)) for line in f]


# Encode labels using LabelEncoder
def encode_labels(doc_data, label_key, label_encoder=None):

    data = {"labels": [], "texts": []}

    all_labels = [label for doc in doc_data for label in doc[label_key]]
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(all_labels)
    for doc in doc_data:
        labels = label_encoder.transform(doc[label_key]).tolist()
        lines = doc["text"].split("\n")
        # Ensure that there is always a labels for line. If not, append with "clean"
        if len(labels) < len(lines):
            labels.extend(
                [label_encoder.transform(["clean"])[0]] * (len(lines) - len(labels))
            )
        # If there are too many labels, truncate the list
        elif len(labels) > len(lines):
            labels = labels[: len(lines)]
        data["labels"].append(labels)
        data["texts"].append(lines)
    return data, label_encoder


# Function to calculate clean vs other ratio for a specific split
def calculate_clean_vs_other_ratio(dataset_dict, split_name, clean_label_index):
    labels = dataset_dict[split_name]["labels"]

    clean_count = 0
    total_count = 0

    # Iterate over all documents in the split
    for doc_labels in labels:
        for label_list in doc_labels:
            total_count += len(label_list)
            clean_count += sum(1 for label in label_list if label == clean_label_index)

    other_count = total_count - clean_count
    clean_ratio = clean_count / total_count if total_count > 0 else 0

    return clean_count, other_count, clean_ratio


# Function to downsample clean documents proportionally
def downsample_clean_documents_proportionally(
    dataset_dict, clean_label_index, target_clean_ratio
):
    split_name = "train"  # Only applying the operation to the "train" split
    data = dataset_dict[split_name]
    texts = data["texts"]
    labels = data["labels"]

    # Count the total number of clean and non-clean labels in the current split
    total_clean_labels = 0
    total_labels = 0
    doc_clean_ratios = []

    # Calculate the proportion of "clean" labels in each document
    for doc_texts, doc_labels in zip(texts, labels):
        clean_count = sum(
            1 for label_list in doc_labels if clean_label_index in label_list
        )
        total_count = len(doc_labels)

        # Store the proportion of clean labels for this document
        doc_clean_ratios.append(
            {
                "texts": doc_texts,
                "labels": doc_labels,
                "clean_count": clean_count,
                "total_count": total_count,
                "clean_ratio": clean_count / total_count if total_count > 0 else 0,
            }
        )

        total_clean_labels += clean_count
        total_labels += total_count

    # Calculate the current clean ratio
    current_clean_ratio = total_clean_labels / total_labels
    if current_clean_ratio <= target_clean_ratio:
        print(
            f"No downsampling needed for {split_name}, current clean ratio: {current_clean_ratio}"
        )
        return dataset_dict  # Skip downsampling if the ratio is already at or below the target

    # Downsample the documents
    clean_docs = [doc for doc in doc_clean_ratios if doc["clean_ratio"] > 0.5]
    other_docs = [doc for doc in doc_clean_ratios if doc["clean_ratio"] <= 0.5]

    # Shuffle the clean documents for random sampling
    random.shuffle(clean_docs)

    # Calculate the number of clean labels to remove to reach the target clean ratio
    target_clean_count = int(total_labels * target_clean_ratio)
    clean_labels_to_remove = total_clean_labels - target_clean_count

    # Track which documents we remove
    removed_clean_docs = []

    for doc in clean_docs:
        if clean_labels_to_remove <= 0:
            break
        clean_labels_to_remove -= doc["clean_count"]
        removed_clean_docs.append(doc)

    # Create the final list of remaining documents
    remaining_docs = [doc for doc in doc_clean_ratios if doc not in removed_clean_docs]

    # Separate texts and labels
    dataset_dict[split_name]["texts"], dataset_dict[split_name]["labels"] = zip(
        *[(doc["texts"], doc["labels"]) for doc in remaining_docs]
    )

    return dataset_dict


# Main function to load and preprocess the data
def get_data(multiclass, downsample_ratio=0.1):

    label_key = "llm_junk_annotations"

    # Load and process datasets
    train_data = load_jsonl("data/train.jsonl", label_key, multiclass)
    test_data = load_jsonl("data/test.jsonl", label_key, multiclass)
    dev_data = load_jsonl("data/dev.jsonl", label_key, multiclass)

    # Encode labels and create Hugging Face datasets
    train_data, label_encoder = encode_labels(train_data, label_key)
    test_data, _ = encode_labels(test_data, label_key, label_encoder)
    dev_data, _ = encode_labels(dev_data, label_key, label_encoder)

    # Find the label index for 'clean'
    clean_label_index = label_encoder.transform(["clean"])[0]

    # Create DatasetDict with all splits
    dataset_dict = {
        "train": train_data,
        "test": test_data,
        "dev": dev_data,
    }

    # First, calculate the initial clean vs other ratio for the "train" split
    initial_clean_count, initial_other_count, initial_clean_ratio = (
        calculate_clean_vs_other_ratio(dataset_dict, "train", clean_label_index)
    )

    print(f"Initial clean count: {initial_clean_count}")
    print(f"Initial other count: {initial_other_count}")
    print(f"Initial clean ratio: {initial_clean_ratio}")

    if downsample_ratio < 1.0:
        dataset_dict = downsample_clean_documents_proportionally(
            dataset_dict, clean_label_index, downsample_ratio
        )

        # Finally, calculate the clean vs other ratio again after downsampling
        final_clean_count, final_other_count, final_clean_ratio = (
            calculate_clean_vs_other_ratio(dataset_dict, "train", clean_label_index)
        )

        print(f"Final clean count: {final_clean_count}")
        print(f"Final other count: {final_other_count}")
        print(f"Final clean ratio: {final_clean_ratio}")

    return dataset_dict, label_encoder
