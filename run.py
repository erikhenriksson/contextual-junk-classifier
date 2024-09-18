import json
from train import run
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="../llm-junklabeling/output/fineweb_annotated_gpt4_multi_2.jsonl",
)
parser.add_argument("--mode", type=str, default="binary")
parser.add_argument("--train", type=str, default="no")
parser.add_argument("--data_source", type=str, default="llm")
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--model_type", type=str, default="contextual-pooling")
parser.add_argument("--line_window", type=int, default=1)
parser.add_argument("--load_checkpoint", type=str, default="checkpoint-1000")
args = parser.parse_args()

# Print the arguments in JSON format
print(json.dumps(vars(args), indent=4))

run(args)
