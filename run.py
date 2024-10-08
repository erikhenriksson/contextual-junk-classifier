import argparse
import json

from train_base_model import run as run_base
from train_hierarchical_model import run as run_hierarchical

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--multiclass", action="store_true")
parser.add_argument(
    "--model", type=str, default="base", choices=["base", "hierarchical"]
)
parser.add_argument("--train", action="store_true")
parser.add_argument("--base_model", type=str, default="xlm-roberta-base")
parser.add_argument("--downsample_clean_ratio", type=float, default=0.1)
parser.add_argument("--use_class_weights", action="store_true")
parser.add_argument("--add_synthetic_data", action="store_true")
parser.add_argument("--use_focal_loss", action="store_true")
parser.add_argument("--freeze_base_model", action="store_true")
parser.add_argument("--embedding_model", action="store_true")
args = parser.parse_args()

# Print the arguments in JSON format
print(json.dumps(vars(args), indent=4))

if args.model == "base":
    run_base(args)
elif args.model == "hierarchical":
    run_hierarchical(args)
