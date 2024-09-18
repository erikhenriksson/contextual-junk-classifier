from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import torch


class ContextualTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context_left = item["context_left"]
        target_text = item["target_text"]
        context_right = item["context_right"]
        label = item["label"]

        # Tokenize target text
        target_encoding = self.tokenizer(
            target_text.strip(),
            add_special_tokens=False,
            truncation=False,
        )
        target_input_ids = target_encoding["input_ids"]
        len_target_text = len(target_input_ids)

        # Check if target text exceeds max_length
        if len_target_text >= self.max_length - 2:  # Account for <s> and </s>
            # Omit context, use truncated target text
            truncated_target_ids = target_input_ids[: self.max_length - 2]
            input_ids = (
                [self.tokenizer.cls_token_id]
                + truncated_target_ids
                + [self.tokenizer.sep_token_id]
            )
            attention_mask = [1] * len(input_ids)
            target_mask = [0] + [1] * len(truncated_target_ids) + [0]
        else:
            # Tokenize contexts
            context_left_encoding = self.tokenizer(
                context_left.strip(),
                add_special_tokens=False,
                truncation=False,
            )
            context_right_encoding = self.tokenizer(
                context_right.strip(),
                add_special_tokens=False,
                truncation=False,
            )
            context_left_ids = context_left_encoding["input_ids"]
            context_right_ids = context_right_encoding["input_ids"]

            # Calculate available space for contexts
            total_available_length = (
                self.max_length - len_target_text - 4
            )  # Account for special tokens
            # The 4 accounts for <s> and </s> around target, and potentially <s> tokens for contexts

            # Initially assign half to each context
            left_available = total_available_length // 2
            right_available = total_available_length - left_available

            # Trim contexts from beginning of left context and end of right context
            if len(context_left_ids) > left_available:
                context_left_ids = context_left_ids[-left_available:]
            if len(context_right_ids) > right_available:
                context_right_ids = context_right_ids[:right_available]

            # Build input_ids
            input_ids = (
                [self.tokenizer.cls_token_id]
                + context_left_ids
                + [self.tokenizer.sep_token_id]
                + target_input_ids
                + [self.tokenizer.sep_token_id]
                + context_right_ids
                + [self.tokenizer.sep_token_id]
            )
            attention_mask = [1] * len(input_ids)

            # Build target_mask
            # Calculate positions of target tokens in input_ids
            # Target tokens start after cls_token and context_left_ids and sep_token
            target_start = 1 + len(context_left_ids) + 1
            target_end = target_start + len_target_text
            target_mask = [0] * len(input_ids)
            for i in range(target_start, target_end):
                if i < len(target_mask):
                    target_mask[i] = 1

        if len(input_ids) > self.max_length:
            print("Warning: input_ids exceed max_length")

            print(len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "labels": label,
        }


class ContextualDataCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        target_masks = [feature["target_mask"] for feature in features]

        # Pad input_ids and attention_masks
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad target_masks
        max_seq_length = batch["input_ids"].shape[1]
        padded_target_masks = torch.zeros(
            (len(target_masks), max_seq_length), dtype=torch.long
        )
        for i, mask in enumerate(target_masks):
            length = len(mask)
            padded_target_masks[i, :length] = torch.tensor(mask, dtype=torch.long)

        batch["target_mask"] = padded_target_masks
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def _create_dataset(source_data, tokenizer, prefix=""):
    data = {
        "text": [f"{prefix}{x['text']}" for x in source_data],
        "label": [x["label"] for x in source_data],
    }

    dataset = HFDataset.from_dict(data)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    # Apply the tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set the format for PyTorch
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    return tokenized_dataset


def create_basic_dataset_with_query_prefix(source_data, tokenizer):
    return _create_dataset(
        source_data,
        tokenizer,
        "Represent this sentence for searching relevant passages: ",
    )


def create_basic_dataset(source_data, tokenizer):
    return _create_dataset(
        source_data,
        tokenizer,
    )
