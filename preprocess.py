import json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import random

random.seed(42)
np.random.seed(42)


def read_json_to_list(file_path, mode):
    def map_annotation(annotation):
        if annotation == "0":
            return "junk"
        return "clean"

    result_list = []
    with open(file_path, "r") as f:
        data = json.loads(f.read())

        for d in data:
            result_list.append(
                (d["text"], [map_annotation(annotation) for annotation in d["labels"]])
            )
    return result_list


def read_jsonl_to_list(file_path, mode):
    def map_annotation(annotation):
        label = annotation.lower().strip()

        if mode == "binary":
            if label != "clean":
                label = "junk"

        return label

    result_list = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            text_lines = data.get("text", "").split("\n")
            llm_junk_annotations = [
                map_annotation(annotation)
                for annotation in data.get("llm_junk_annotations", [])
            ]
            result_list.append((text_lines, llm_junk_annotations))
    return result_list


def get_unique_sorted_labels(parsed_data):
    unique_labels = set()
    label_counts = Counter()

    for _, annotations in parsed_data:
        for annotation in annotations:
            unique_labels.add(annotation)
            label_counts[annotation] += 1

    sorted_labels = sorted(unique_labels)
    return sorted_labels, label_counts


def encode_labels(parsed_data, unique_labels):
    updated_data = []

    for text_lines, annotations in parsed_data:
        encoded_labels = [unique_labels.index(x) for x in annotations]

        # Append the tuple with text_lines and encoded labels
        updated_data.append((text_lines, encoded_labels))

    return updated_data


def create_context_windows(docs, base_label_idx, window_size=1):
    windowed_docs = []
    for doc in docs:

        text = doc[0]
        label = doc[1]
        context_windows = []
        num_lines = len(text)

        for i in range(num_lines):
            start = max(0, i - window_size)
            end = min(num_lines, i + window_size + 1)
            try:
                this_label = label[i]
            except:
                print("Warning: label not found, using 'clean' as default")
                this_label = base_label_idx
            window = {
                "context_left": "\n".join(text[start:i]),
                "target_text": text[i],
                "context_right": "\n".join(text[i + 1 : end]),
                "label": this_label,
            }
            context_windows.append(window)
        windowed_docs += context_windows
    print(len(windowed_docs))
    return windowed_docs


def downsample_class(data, class_label, downsample_ratio=0.3):
    if downsample_ratio >= 1.0:
        return data
    target_class_data = [item for item in data if item["label"] == class_label]
    other_class_data = [item for item in data if item["label"] != class_label]

    retain_count = int(len(target_class_data) * downsample_ratio)
    downsampled_class_data = random.sample(target_class_data, retain_count)

    return downsampled_class_data + other_class_data


def get_data(
    file_path, source_type="jsonl", mode="binary", downsample_ratio=0.3, window_size=1
):

    func = read_jsonl_to_list if source_type == "jsonl" else read_json_to_list
    parsed_data = func(file_path, mode)
    unique_sorted_labels, counts = get_unique_sorted_labels(parsed_data)

    print("Unique labels:", unique_sorted_labels)
    print("Label counts:", counts)

    data = encode_labels(parsed_data, unique_sorted_labels)

    # Step 1: Compute the most frequent label for each document
    def most_frequent_label(labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[
            np.argmax(counts)
        ]  # Return the label with the highest count

    # Compute the most frequent label for stratification
    most_frequent_labels = [most_frequent_label(labels) for _, labels in data]

    # Step 2: First split into train (70%) and temp (30%)
    data_train, data_temp, _, labels_temp = train_test_split(
        data,
        most_frequent_labels,
        test_size=0.3,
        stratify=most_frequent_labels,
        random_state=42,
    )

    # Step 3: Split temp into dev (10%) and test (20%)
    data_dev, data_test, _, _ = train_test_split(
        data_temp, labels_temp, test_size=2 / 3, stratify=labels_temp, random_state=42
    )

    # Function to count label distributions in a split
    def get_label_distribution(split_data):
        all_labels = []
        for _, labels in split_data:
            all_labels.extend(
                labels
            )  # Collect all labels from the lines in each document
        return Counter(all_labels)

    # Step 4: Print label distributions for each split
    train_distribution = get_label_distribution(data_train)
    dev_distribution = get_label_distribution(data_dev)
    test_distribution = get_label_distribution(data_test)

    print("Label distribution in train split:", train_distribution)
    print("Label distribution in dev split:", dev_distribution)
    print("Label distribution in test split:", test_distribution)

    # Check for data leakage: Ensure there are no overlapping documents or labels across the splits
    def get_unique_documents(data_split):
        return set([tuple(lines) for lines, _ in data_split])

    # Get unique documents for each split
    train_docs = get_unique_documents(data_train)
    dev_docs = get_unique_documents(data_dev)
    test_docs = get_unique_documents(data_test)

    # Helper function to calculate overlap and percentage leakage
    def calculate_leakage_percentage(split_a, split_b, name_a, name_b):
        overlap = split_a.intersection(split_b)
        overlap_count = len(overlap)
        leakage_a = (overlap_count / len(split_a)) * 100 if len(split_a) > 0 else 0
        leakage_b = (overlap_count / len(split_b)) * 100 if len(split_b) > 0 else 0
        print(f"Overlap between {name_a} and {name_b}: {overlap_count} documents")
        print(f"Leakage percentage for {name_a}: {leakage_a:.2f}%")
        print(f"Leakage percentage for {name_b}: {leakage_b:.2f}%")
        return overlap_count

    # Calculate the extent of leakage between each split
    leakage_train_dev = calculate_leakage_percentage(
        train_docs, dev_docs, "Train", "Dev"
    )
    leakage_train_test = calculate_leakage_percentage(
        train_docs, test_docs, "Train", "Test"
    )
    leakage_dev_test = calculate_leakage_percentage(dev_docs, test_docs, "Dev", "Test")

    # If no overlap is found, there is no leakage
    if leakage_train_dev == 0 and leakage_train_test == 0 and leakage_dev_test == 0:
        print("No data leakage detected.")
    else:
        print("Data leakage detected. See above for details.")

    base_label_idx = unique_sorted_labels.index("clean")

    data_train = downsample_class(
        create_context_windows(
            data_train,
            base_label_idx,
            window_size=window_size,
        ),
        base_label_idx,
        downsample_ratio,
    )
    data_test = downsample_class(
        create_context_windows(
            data_test,
            base_label_idx,
            window_size=window_size,
        ),
        base_label_idx,
        downsample_ratio,
    )
    data_dev = downsample_class(
        create_context_windows(
            data_dev,
            base_label_idx,
            window_size=window_size,
        ),
        base_label_idx,
        downsample_ratio,
    )
    print("after downsampling")
    print(len(data_train), len(data_test), len(data_dev))

    return data_train, data_dev, data_test, len(unique_sorted_labels)
