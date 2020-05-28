import argparse
import sys
import os

path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")
sys.path.insert(0, path)

from pyrec.dataset import Dataset
from pyrec.builtin_datasets import BUILTIN_DATASETS

class MyParser(argparse.ArgumentParser):
    """
    A parser which prints the help message when an error occurs.
    """
    def error(self, message):
       sys.stderr.write(f"error: {message} \n")
       self.print_help()
       sys.exit(2)

def main():
    parser = MyParser(
        description="Evaluate the performance of a rating prediction",
    )

    parser.add_argument(
        "-load-builtin",
        type=str,
        dest="load_builtin",
        help=f"The name of the built-in dataset to use. Allowed values are {','.join(BUILTIN_DATASETS.keys())}. Default is ml-100k",
        default="ml-100k"
    )

    args = parser.parse_args()
    Dataset.load_builtin(args.load_builtin)

if __name__ == '__main__':
    main()
