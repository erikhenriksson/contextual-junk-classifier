from train import run
import sys

data_path = sys.argv[1]
mode = sys.argv[2]  # binary or multiclass
do_train = sys.argv[3] == "train" if len(sys.argv) > 3 else False
run(data_path, mode, do_train)
