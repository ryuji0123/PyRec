
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