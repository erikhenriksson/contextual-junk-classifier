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
args = parser.parse_args()

run(args)
