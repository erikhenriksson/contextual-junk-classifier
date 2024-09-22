import json
from collections import defaultdict


# Load the data and dynamically build the label-to-index mapping
def build_label_mapping(file_path):
    label_to_index = {"clean": 0}  # We can fix "clean" to index 0
    current_index = 1  # Start indexing other labels from 1

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())  # Load each JSON line
            annotation_labels = [
                x.lower() for x in data["llm_junk_annotations"]
            ]  # Extract the labels

            # Dynamically add new labels to the mapping
            for label in annotation_labels:
                if label not in label_to_index:
                    label_to_index[label] = current_index
                    current_index += 1

    return label_to_index


# Load and process the data using the dynamically created label mapping
def process_data(file_path, label_to_index):
    documents = []
    labels = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())  # Load each JSON line
            text = data["text"]  # Extract the text field
            annotation_labels = [
                x.lower() for x in data["llm_junk_annotations"]
            ]  # Extract the labels

            # Split the text into lines based on newline characters
            lines = text.split("\n")

            # Convert annotations to integer indices
            annotation_indices = [label_to_index[label] for label in annotation_labels]

            # Pad the annotation list with "clean" if it's shorter than the number of lines
            while len(annotation_indices) < len(lines):
                annotation_indices.append(
                    label_to_index["clean"]
                )  # Pad with "clean" index

            # Add processed data to lists
            documents.append(lines)  # Append the lines (document) to the list
            labels.append(annotation_indices)  # Append the padded annotations

    return documents, labels


def process_eval_data(file_path):
    documents = []
    labels = []
    with open(file_path, "r") as f:
        data = json.loads(f.read())

        for d in data:
            documents.append(d["text"])
            labels.append([int(not (int(x))) for x in d["labels"]])

    return documents, labels
