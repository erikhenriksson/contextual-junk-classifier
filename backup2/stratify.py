import json
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split

taxonomy = [
    "boilerplate",
    "clean",
    "code",
    "data",
    "junk",
    "metadata",
    "navigational",
    "non-english",
    "noise",
]
# Path to the JSONL file
file_path = "../llm-junklabeling/output_fixed3.jsonl"

# Initialize an empty list to store the data
data = []


def map_label(label):
    label = label.lower()
    # if label in ["noise", "other junk", "code"]:
    #    label = "other junk"
    # elif label not in taxonomy:
    #    label = "clean"
    return label


# Open the JSONL file and read each line
with open(file_path, "r") as f:
    for line in f:
        # Parse each line as a JSON object
        line = json.loads(line)
        line["llm_junk_annotations"] = [
            map_label(label) for label in line["llm_junk_annotations"]
        ]
        data.append(line)

print("Number of documents:", len(data))

# Extract the labels from each document
all_labels = [label for doc in data for label in doc["llm_junk_annotations"]]

# Count the occurrences of each label
label_counts = Counter(all_labels)

# Get unique labels
unique_labels = set(all_labels)

# Get number of unique labels
n_labels = len(unique_labels)

# Sort the counts by label names (keys)
sorted_label_counts = dict(
    sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
)

print("Number of labels:", len(all_labels))
print("Number of unique labels:", len(label_counts))
print("Label counts:", sorted_label_counts)


# Initialize LabelEncoder and fit on the unique labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform string labels into integers for each document
for doc in data:
    doc["encoded_labels"] = label_encoder.transform(doc["llm_junk_annotations"])

# Extract the labels from each document
all_labels_encoded = [doc["encoded_labels"] for doc in data]


binary_labels = []

for labels in all_labels_encoded:
    doc_label = [0] * n_labels
    for label in labels:
        doc_label[label - 1] = 1  # Mark presence of the label
    binary_labels.append(doc_label)

binary_labels = np.array(binary_labels)

X = np.array(data)


# Step 1: Perform 80/20 split using IterativeStratification
stratifier = IterativeStratification(
    n_splits=2, order=1, sample_distribution_per_fold=[0.8, 0.2]
)

remaining_indices, train_indices = next(stratifier.split(X, binary_labels))

X_train, X_remaining = X[train_indices], X[remaining_indices]
y_train, y_remaining = binary_labels[train_indices], binary_labels[remaining_indices]

# Step 2: Split the remaining 20% into test (10%) and dev (10%)
stratifier_remaining = IterativeStratification(
    n_splits=2, order=1, sample_distribution_per_fold=[0.5, 0.5]
)

dev_indices, test_indices = next(stratifier_remaining.split(X_remaining, y_remaining))

train_set = set(train_indices)
test_set = set(remaining_indices[test_indices])
dev_set = set(remaining_indices[dev_indices])

# Check if there is any overlap between the sets
train_test_intersection = train_set.intersection(test_set)
train_dev_intersection = train_set.intersection(dev_set)
test_dev_intersection = test_set.intersection(dev_set)

# Print the results
print("Train-Test Intersection:", train_test_intersection)
print("Train-Dev Intersection:", train_dev_intersection)
print("Test-Dev Intersection:", test_dev_intersection)
print()

X_test, X_dev = X_remaining[test_indices], X_remaining[dev_indices]
y_test, y_dev = y_remaining[test_indices], y_remaining[dev_indices]

# Results:
print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Dev size:", len(X_dev))


# Function to print label distribution
def print_label_distribution(labels, split_name):
    label_distribution = np.sum(labels, axis=0)
    print(f"{split_name} Label Distribution:")
    for i, count in enumerate(label_distribution, 1):
        print(f"Label {i}: {count}")
    print("\n")


# Print label distributions
print_label_distribution(y_train, "Train")
print_label_distribution(y_test, "Test")
print_label_distribution(y_dev, "Dev")


# Save the split data to JSONL format
def save_jsonl(data, filename):
    with open(filename, "w") as outfile:
        for entry in data:
            del entry["encoded_labels"]
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


# Save the train, test, and dev splits
save_jsonl(X_train, "data/train.jsonl")
save_jsonl(X_test, "data/test.jsonl")
save_jsonl(X_dev, "data/dev.jsonl")

print("Data splits saved successfully!")
