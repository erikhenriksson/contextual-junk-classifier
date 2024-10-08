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
        # for i, label in enumerate(labels):
        #    if label in ["noise", "other junk", "code"]:
        #        labels[i] = "other junk"
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
        # Count the number of clean labels in the document
        total_count += len(doc_labels)
        clean_count += sum(1 for label in doc_labels if label == clean_label_index)

    other_count = total_count - clean_count
    clean_ratio = clean_count / total_count if total_count > 0 else 0

    return clean_count, other_count, clean_ratio


# Function to downsample documents with 100% clean labels
def downsample_documents_with_all_clean_labels(
    dataset_dict, clean_label_index, target_clean_ratio
):
    split_name = "train"  # Only applying the operation to the "train" split
    data = dataset_dict[split_name]
    texts = data["texts"]
    labels = data["labels"]

    # Count the total number of clean and non-clean labels in the current split
    total_clean_labels = 0
    total_labels = 0

    # Identify documents that are 100% clean
    all_clean_docs = []
    mixed_docs = []

    for doc_texts, doc_labels in zip(texts, labels):
        clean_count = sum(1 for label in doc_labels if label == clean_label_index)
        total_count = len(doc_labels)

        total_clean_labels += clean_count
        total_labels += total_count

        # If all labels in the document are clean, mark it as all_clean
        if clean_count == total_count:
            all_clean_docs.append((doc_texts, doc_labels))
        else:
            mixed_docs.append((doc_texts, doc_labels))

    # Calculate the current clean ratio
    current_clean_ratio = total_clean_labels / total_labels
    if current_clean_ratio <= target_clean_ratio:
        print(
            f"No downsampling needed for {split_name}, current clean ratio: {current_clean_ratio}"
        )
        return dataset_dict  # Skip downsampling if the ratio is already at or below the target

    # Calculate the number of clean labels to remove to reach the target clean ratio
    target_clean_count = int(total_labels * target_clean_ratio)
    clean_labels_to_remove = total_clean_labels - target_clean_count

    # Shuffle the all-clean documents for random sampling
    random.shuffle(all_clean_docs)

    # Track which documents to remove
    removed_clean_docs = []

    for doc_texts, doc_labels in all_clean_docs:
        if clean_labels_to_remove <= 0:
            break
        clean_labels_to_remove -= len(
            doc_labels
        )  # Each doc in all_clean_docs is 100% clean
        removed_clean_docs.append((doc_texts, doc_labels))

    # Combine remaining documents (mixed and those all_clean_docs that were not removed)
    remaining_docs = mixed_docs + [
        doc for doc in all_clean_docs if doc not in removed_clean_docs
    ]

    # Separate texts and labels for the remaining documents
    dataset_dict[split_name]["texts"], dataset_dict[split_name]["labels"] = zip(
        *remaining_docs
    )

    return dataset_dict


# Function to remove documents from the training split that have 100% clean labels
def remove_all_clean_documents(dataset_dict, clean_label_index):
    split_name = "train"
    data = dataset_dict[split_name]
    texts = data["texts"]
    labels = data["labels"]

    # Identify documents that are 100% clean
    mixed_docs = []

    for doc_texts, doc_labels in zip(texts, labels):
        clean_count = sum(1 for label in doc_labels if label == clean_label_index)
        total_count = len(doc_labels)

        # If all labels in the document are clean, mark it as all_clean
        if clean_count < total_count:
            mixed_docs.append((doc_texts, doc_labels))

    # Separate texts and labels for the remaining documents
    dataset_dict[split_name]["texts"], dataset_dict[split_name]["labels"] = zip(
        *mixed_docs
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
        dataset_dict = remove_all_clean_documents(dataset_dict, clean_label_index)
        dataset_dict = remove_all_clean_documents(
            dataset_dict, label_encoder.transform(["metadata"])[0]
        )

        # Finally, calculate the clean vs other ratio again after downsampling
        final_clean_count, final_other_count, final_clean_ratio = (
            calculate_clean_vs_other_ratio(dataset_dict, "train", clean_label_index)
        )

        print(f"Final clean count: {final_clean_count}")
        print(f"Final other count: {final_other_count}")
        print(f"Final clean ratio: {final_clean_ratio}")

    return dataset_dict, label_encoder
