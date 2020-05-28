import os
import sys

from .builtin_datasets import BUILTIN_DATASETS, download_builtin_dataset


class Dataset:

    @classmethod
    def load_builtin(cls, name="ml-100k", prompt=True):
        """
        Load a built-in dataset
        :param name(:obj:`string`): The name of the built-in dataset to load
        :param prompt(:obj:`bool`): Prompt before downloading if dataset is not already on disk
        :return:
        """

        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError(f"unknown dataset {name}. Accepted values are {join(BUILTIN_DATASETS.keys())}.")

        if not os.path.isfile(dataset.path):
            answered = not prompt
            while not answered:
                print(f"Dataset {name} could not be found. Do you want to download it [y/n]", end="")
                choice = input().lower()

                if choice in ["yes", "y"]:
                    answered = True
                elif choice in ["no", "n"]:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()
            download_builtin_dataset(name)

