import jsonlines

# Define annotation mapping
ANNOTATION_MAP = {
    "c": "clean",
    "b": "boilerplate",
    "m": "metadata",
    "n": "navigational",
    "s": "structured",
    "x": "code",
    "z": "noise",
    "": "other junk",  # Enter key for 'other junk'
}

# Path to the JSONL file
JSONL_PATH = "data/dev.jsonl"


def annotate_lines():
    # Load the entire file content into memory
    with jsonlines.open(JSONL_PATH) as reader:
        data = list(reader)  # Load the entire file into a list

    # Find the first document that has not been fully annotated
    for i, obj in enumerate(data):
        if (
            "llm_junk_annotations_fixed" in obj
            or not "other junk" in obj["llm_junk_annotations"]
        ):
            continue  # Skip already annotated documents

        # Start annotation process for this document
        print(i)
        text_lines = obj["text"].split("\n")
        annotations = obj["llm_junk_annotations"]
        fixed_annotations = []
        changed = False
        for line, annotation in zip(text_lines, annotations):
            if annotation == "other junk":
                print(f"\nLine: {line}")
                # Ask user for new annotation
                while True:
                    user_input = input(
                        "Enter new annotation (c: clean, b: boilerplate, m: metadata, n: navigational, "
                        "s: structured, x: code, z: noise, [Enter]: other junk): "
                    ).lower()
                    if user_input in ANNOTATION_MAP:
                        fixed_annotations.append(ANNOTATION_MAP[user_input])
                        break
                    else:
                        print("Invalid input. Please try again.")

                changed = True
            else:
                # No change if not "other junk"
                fixed_annotations.append(annotation)

        if changed:
            # Add fixed annotations to the document
            obj["llm_junk_annotations_fixed"] = fixed_annotations

            # Rewrite the entire file with the updated data
            with jsonlines.open(JSONL_PATH, mode="w") as writer:
                writer.write_all(data)  # Write the whole dataset back after each change

            # Break after annotating one document and saving progress
            print(f"Document {i + 1} annotated and saved. Continuing next time...")
            # break  # Save progress after one document


if __name__ == "__main__":
    annotate_lines()
