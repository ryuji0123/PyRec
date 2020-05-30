import numpy as np

class Trainset:
    """
    A train set contains all useful data that constitute a training set.
    """

    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale, raw2inner_id_users, raw2inner_id_items):
        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None

    def __repr__(self):
        return f"n_users: {self.n_users}, n_items: {self.n_items}, n_ratings: {self.n_ratings}"

    def all_ratings(self):
        """
        Generator function to iterate over all ratings.
        :return:
        """
        for u, u_ratings in self.ur.items():
            for i, r in u_ratings:
                yield u, i, r

    def all_users(self):
        """
        Generator functionto iterate over all users
        :return:
        """
        return range(self.n_users)

    def all_items(self):
        """
        Generator function to iterate over all items
        :return:
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        """
        :return: The mean of all ratings.
        """
        if self._global_mean is None:
            self._global_mean = np.mean(
                [r for (_, _, r) in self.all_ratings()]
            )
        return self._global_mean