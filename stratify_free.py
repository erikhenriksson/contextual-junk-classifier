import json
import random
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Load original data
file_path = "../llm-junklabeling/output_mapped.jsonl"
data = []
random_state = 42
random.seed(random_state)

with open(file_path, "r") as f:
    for line in f:
        row = json.loads(line)

        for item in row["content"]:
            data.append(item)

# Downsample the "clean" class with a ratio of 0.25
clean_data = [item for item in data if item["label"] == "clean"]
junk_data = [item for item in data if item["label"] != "clean"]
clean_data = random.sample(clean_data, int(len(clean_data) * 0.3))
data = clean_data + junk_data

# Stratify and split data
train_data, temp_data = train_test_split(
    data,
    test_size=0.3,
    stratify=[item["label"] for item in data],
    random_state=random_state,
)
test_data, dev_data = train_test_split(
    temp_data,
    test_size=1 / 3,
    stratify=[item["label"] for item in temp_data],
    random_state=random_state,
)

# Save splits
output_paths = {
    "train": "data/free_train.jsonl",
    "test": "data/free_test.jsonl",
    "dev": "data/free_dev.jsonl",
}

for split, split_data in zip(output_paths.keys(), [train_data, test_data, dev_data]):
    with open(output_paths[split], "w") as f:
        for item in split_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# Return a summary of the split counts
split_counts = {
    split: len(split_data)
    for split, split_data in zip(output_paths.keys(), [train_data, test_data, dev_data])
}
print("Overall Split Counts:", split_counts)

# Count labels in each split
split_label_counts = {}
for split, split_data in zip(output_paths.keys(), [train_data, test_data, dev_data]):
    label_counts = defaultdict(int)
    for item in split_data:
        label_counts[item["label"]] += 1
    split_label_counts[split] = dict(label_counts)

# Print detailed label counts for each split
print("\nDetailed Label Counts for Each Split:")
for split, counts in split_label_counts.items():
    print(f"{split}:", counts)
