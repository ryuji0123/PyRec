import argparse
import sys
import os
import shutil

path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")
sys.path.insert(0, path)

from pyrec.dataset import Dataset
from pyrec.builtin_datasets import BUILTIN_DATASETS, get_dataset_dir

from pyrec.model_selection import KFold, cross_validate

from pyrec.prediction_algorithms import SVD

class MyParser(argparse.ArgumentParser):
    """
    A parser which prints the help message when an error occurs.
    """
    def error(self, message):
       sys.stderr.write(f"error: {message} \n")
       self.print_help()
       sys.exit(2)

def main():
    algo_choices = {
        'SVD': SVD,
    }

    parser = MyParser(
        description="Evaluate the performance of a rating prediction",
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=algo_choices,
        help=f"The prediction algorithm to use. Allowed values are {','.join(algo_choices.keys())}.",
        metavar="<prediction algorithm>"
    )

    parser.add_argument(
        "--load-builtin",
        type=str,
        dest="load_builtin",
        help=f"The name of the built-in dataset to use. Allowed values are {','.join(BUILTIN_DATASETS.keys())}. Default is ml-100k",
        default="ml-100k"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        dest="clean",
        help=f"Remove the {get_dataset_dir()} directory and exit.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        dest="seed",
        help="The seed to use"
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        dest="n_splits",
        help="The number of folds for cross-validation. Default is 5."
    )

    args = parser.parse_args()

    if args.clean:
        folder_path = get_dataset_dir()
        shutil.rmtree(folder_path)
        print(f"Removed {folder_path}")
        exit()

    data = Dataset.load_builtin(args.load_builtin)
    cv = KFold(n_splits=args.n_splits, random_state=args.seed)
    cross_validate()


if __name__ == '__main__':
    main()
