import numpy as np
from itertools import chain
from ..utils import get_rng

class KFold():
    """
    A basic cross-validation iterator.
    """
    def __init__(self, n_splits=5, random_state=None, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, data):
        """
        Generator function to iterate over
        :param data (:obj: Dataset):
        :return:
        """
        indices = np.arange(len(data))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [data.raw_ratings[i] for i in chain(indices[:start], indices[stop:])]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_splits(self):
        return self.n_splits
