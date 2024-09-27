import json
from train_base_model import run as run_base
from train_hierarchical_model import run as run_hierarchical
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--multiclass", action="store_true")
parser.add_argument("--train_base_model", action="store_true")
parser.add_argument("--train_hierarchical_model", action="store_true")
parser.add_argument("--predict_base_model", action="store_true")
parser.add_argument("--predict_hierarchical_model", action="store_true")
parser.add_argument("--base_model", type=str, default="xlm-roberta-base")
args = parser.parse_args()

# Print the arguments in JSON format
print(json.dumps(vars(args), indent=4))

if args.train_base_model:
    run_base(args)
if args.train_hierarchical_model:
    run_hierarchical(args)

if args.predict_base_model:
    run_base(args, just_predict=True)
if args.predict_hierarchical_model:
    run_hierarchical(args, just_predict=True)
