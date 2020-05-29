import os
import sys
import itertools

from .builtin_datasets import BUILTIN_DATASETS, download_builtin_dataset
from .reader import Reader

class Dataset:
    def __init__(self, reader):
        self.reader = reader

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
                    print("Ok then, I'm out!")
                    sys.exit()
            download_builtin_dataset(name)

        reader = Reader(**dataset.reader_params)
        return cls.load_from_file(file_path=dataset.path, reader=reader)

    @classmethod
    def load_from_file(cls, file_path, reader):
        """
        Load a dataset from a (custom) file
        :param file_path:
        :param reader:
        :return:
        """
        return DatasetAutoFolds(rating_file=file_path, reader=reader)

    def read_ratings(self, file_name):
        """
        :param file_name:
        :return: a list of ratings (user, item, rating, timestamp) read from file_name
        """
        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [
                self.reader.parse_line(line) for line in itertools.islice(f, self.reader.skip_lines, None)
            ]

        return raw_ratings

class DatasetAutoFolds(Dataset):
    """
    A derived class from :class: `Dataset`
    """
    def __init__(self, rating_file=None, reader=None, df=None):
        Dataset.__init__(self, reader)
        self.has_been_split = False

        if rating_file is not None:
            self.rating_file = rating_file
            self.raw_ratings = self.read_ratings(self.rating_file)
        elif df is not None:
            pass
        else:
            raise ValueError("Must specify ratings file or dataframe.")

    def __len__(self):
        return len(self.raw_ratings)

    def __str__(self):
        return "\n".join([",".join(list(map(str, rating_line))) for rating_line in self.raw_ratings])
