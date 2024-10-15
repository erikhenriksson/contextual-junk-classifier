import json


splits = ["train", "test", "dev"]

for split in splits:

    # Open and read the JSONL file line by line
    with open(f"data/{split}.jsonl", "r") as f:
        for line in f:
            # Parse each line as a JSON object
            data = json.loads(line)

            # Extract the "text" and "llm_junk_annotations"
            text = data.get("text", "")
            annotations = data.get("llm_junk_annotations", [])

            # Split the text into individual lines
            lines = text.split("\n")

            # Check for each annotation if it's "Other Junk"
            for idx, annotation in enumerate(annotations):
                if annotation == "other junk" and idx < len(lines):
                    print(lines[idx])
