import argparse
import json

from train_base_model import run as run_base
from train_hierarchical_model import run as run_hierarchical
from predict_platt import run as run_platt

# from train_embedding_classifier import run as run_embedding

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--multiclass", action="store_true")
parser.add_argument(
    "--model",
    type=str,
    default="base",
    choices=["base", "hierarchical", "embedding_classifier", "platt"],
)
parser.add_argument("--train", action="store_true")
parser.add_argument("--base_model", type=str, default="xlm-roberta-base")
parser.add_argument("--local_model", type=str, default="")
parser.add_argument("--downsample_clean_ratio", type=float, default=0.1)
parser.add_argument("--use_class_weights", action="store_true")
parser.add_argument("--add_synthetic_data", action="store_true")
parser.add_argument("--use_focal_loss", action="store_true")
parser.add_argument("--freeze_base_model", action="store_true")
parser.add_argument("--embedding_model", action="store_true")
parser.add_argument("--n_dim", type=int, default=768)
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--patience", type=int, default=5)
args = parser.parse_args()

# Print the arguments in JSON format
print(json.dumps(vars(args), indent=4))

if args.model == "base":
    run_base(args)
elif args.model == "hierarchical":
    run_hierarchical(args)
elif args.model == "platt":
    run_platt(args)