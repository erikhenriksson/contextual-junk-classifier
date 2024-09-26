import json
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict


# Load JSONL data from file
def load_jsonl(filename, label_key, multiclass):
    def convert(line):
        labels = line[label_key]
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
def create_hf_dataset(data, label_key):
    return Dataset.from_dict(
        {
            key if key != label_key else "labels": [doc[key] for doc in data]
            for key in data[0].keys()
        }
    )


def get_data(multiclass):

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
    return dataset_dict, label_encoder
