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
            return 1
        return 0

    result_list = []
    with open(file_path, "r") as f:
        data = json.loads(f.read())

        for d in data:
            result_list.append(
                (d["text"], [map_annotation(annotation) for annotation in d["labels"]])
            )
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


def create_context_window(doc, base_label_idx, unique_sorted_labels, window_size):
    text = doc[0]
    label = [unique_sorted_labels.index(x) for x in doc[1]]
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

    return context_windows


def get_eval_data(file_path, mode, split=False):

    parsed_data = read_json_to_list(file_path, mode)
    unique_sorted_labels, counts = get_unique_sorted_labels(parsed_data)
    base_label_idx = unique_sorted_labels.index(0)
    print(unique_sorted_labels)
    print(counts)
    print(parsed_data[0])
    texts = []
    for doc in parsed_data:
        texts += create_context_window(
            doc, base_label_idx, unique_sorted_labels, window_size=1
        )
    if not split:
        return np.array(texts)

    else:

        def get_label_distribution(dataset):
            labels = [example["label"] for example in dataset]
            return Counter(labels)

        labels = np.array([item["label"] for item in texts])

        train_data, dev_test_data = train_test_split(
            texts, test_size=0.3, stratify=labels, random_state=42
        )
        dev_test_labels = [item["label"] for item in dev_test_data]
        dev_data, test_data = train_test_split(
            dev_test_data, test_size=2 / 3, stratify=dev_test_labels, random_state=42
        )

        print(f"Train size: {len(train_data)}")
        print(f"Dev size: {len(dev_data)}")
        print(f"Test size: {len(test_data)}")

        # Get label distributions for each split
        train_distribution = get_label_distribution(train_data)
        dev_distribution = get_label_distribution(dev_data)
        test_distribution = get_label_distribution(test_data)

        # Print distributions to inspect
        print("Train Label Distribution:", train_distribution)
        print("Dev Label Distribution:", dev_distribution)
        print("Test Label Distribution:", test_distribution)

        print(train_data[0])

        return train_data, dev_data, test_data, len(unique_sorted_labels)
