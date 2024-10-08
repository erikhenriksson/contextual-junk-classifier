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


# Function to remove documents from the training split that contain only specified labels
def remove_specified_label_only_documents(dataset_dict, excluded_label_indexes):
    split_name = "train"
    data = dataset_dict[split_name]
    texts = data["texts"]
    labels = data["labels"]

    # Identify documents that contain labels other than the specified ones
    mixed_docs = []

    for doc_texts, doc_labels in zip(texts, labels):
        # Check if all labels in the document are within the excluded_label_indexes list
        if any(label not in excluded_label_indexes for label in doc_labels):
            # If there is at least one label not in the excluded list, keep the document
            mixed_docs.append((doc_texts, doc_labels))

    # Separate texts and labels for the remaining documents
    if mixed_docs:  # Check if there are documents left after filtering
        dataset_dict[split_name]["texts"], dataset_dict[split_name]["labels"] = zip(
            *mixed_docs
        )
    else:
        # If no documents remain, empty the texts and labels lists
        dataset_dict[split_name]["texts"], dataset_dict[split_name]["labels"] = [], []

    return dataset_dict


# Function to chunk the training split with a minimum chunk size of 5
def chunk_training_split(dataset_dict, chunk_size=16, min_chunk_size=5):
    split_name = "train"
    data = dataset_dict[split_name]
    texts = data["texts"]
    labels = data["labels"]
    chunked_texts = []
    chunked_labels = []

    for doc_texts, doc_labels in zip(texts, labels):
        # Initialize temporary storage for chunks
        doc_chunks_text = []
        doc_chunks_label = []

        # Split the document into chunks of size chunk_size
        for i in range(0, len(doc_texts), chunk_size):
            chunk_text = doc_texts[i : i + chunk_size]
            chunk_label = doc_labels[i : i + chunk_size]

            # If this is the last chunk and it is smaller than min_chunk_size, append to the previous chunk
            if len(chunk_text) < min_chunk_size and doc_chunks_text:
                # Append the last chunk to the previous chunk
                doc_chunks_text[-1].extend(chunk_text)
                doc_chunks_label[-1].extend(chunk_label)
            else:
                # Otherwise, add as a new chunk
                doc_chunks_text.append(chunk_text)
                doc_chunks_label.append(chunk_label)

        # Append the document chunks to the main list
        chunked_texts.extend(doc_chunks_text)
        chunked_labels.extend(doc_chunks_label)

    # Update the dataset dictionary
    dataset_dict[split_name]["texts"], dataset_dict[split_name]["labels"] = (
        chunked_texts,
        chunked_labels,
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

    # Chunk the training split to chunks of 20 lines

    # Create DatasetDict with all splits
    dataset_dict = {
        "train": train_data,
        "test": test_data,
        "dev": dev_data,
    }

    dataset_dict = chunk_training_split(dataset_dict)

    # First, calculate the initial clean vs other ratio for the "train" split
    initial_clean_count, initial_other_count, initial_clean_ratio = (
        calculate_clean_vs_other_ratio(dataset_dict, "train", clean_label_index)
    )

    print(f"Initial clean count: {initial_clean_count}")
    print(f"Initial other count: {initial_other_count}")
    print(f"Initial clean ratio: {initial_clean_ratio}")

    if downsample_ratio < 1.0:
        dataset_dict = remove_specified_label_only_documents(
            dataset_dict, [clean_label_index]
        )

        # Finally, calculate the clean vs other ratio again after downsampling
        final_clean_count, final_other_count, final_clean_ratio = (
            calculate_clean_vs_other_ratio(dataset_dict, "train", clean_label_index)
        )

        print(f"Final clean count: {final_clean_count}")
        print(f"Final other count: {final_other_count}")
        print(f"Final clean ratio: {final_clean_ratio}")

    return dataset_dict, label_encoder
