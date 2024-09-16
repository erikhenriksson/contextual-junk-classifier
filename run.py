from trainer import run
import sys

data_path = sys.argv[1]
mode = sys.argv[2]  # binary or multiclass

run(data_path, mode)
