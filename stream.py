from datasets import load_from_disk


# Create a generator function that yields rows from all shards
def stream_from_shards(shard_dirs):
    for shard_dir in shard_dirs:
        # Load the dataset shard from disk
        dataset = load_from_disk(shard_dir)

        # Iterate over rows in the dataset and yield one at a time
        for row in dataset:
            yield row


NUM_SHARDS = 3
# List of shard directories (assuming shard_0, shard_1, etc.)
shard_dirs = [
    f"exquisiteweb/shard_{i}" for i in range(NUM_SHARDS)
]  # Replace NUM_SHARDS with the actual number

# Create the generator
data_iterator = stream_from_shards(shard_dirs)

# Example of how to use the iterator to get the next row
for row in data_iterator:
    # Process the row (this can be whatever you need to do with each row)
    print(row)  # You can replace this with your actual processing logic
