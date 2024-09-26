import json
from train_base_model import run
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--multiclass", action="store_true")
parser.add_argument("--train_base_model", action="store_true")
parser.add_argument("--train_final_model", action="store_true")
parser.add_argument("--base_model", type=str, default="xlm-roberta-base")
args = parser.parse_args()

# Print the arguments in JSON format
print(json.dumps(vars(args), indent=4))

run(args)
