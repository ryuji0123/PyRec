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

        self._inner2raw_id_users = None
        self._inner2raw_id_items = None

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

    def to_inner_uid(self, raw_uid):
        """
        Convert a user raw id to an inner id.
        :param raw_uid:
        :return:
        """
        try:
            return self._raw2inner_id_users[raw_uid]
        except KeyError:
            raise ValueError(
                f"User {str(raw_uid)} is not part of the trainset."
            )

    def to_raw_uid(self, iuid):
        """
        Convert a user inner id to a raw id
        :param iuid:
        :return:
        """
        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {
                inner: raw for (raw, inner) in
                self._raw2inner_id_users.items()
            }
        try:
            return self._inner2raw_id_users[iuid]
        except KeyError:
            raise ValueError(f"{str(iuid)} is not a valid inner id.")

    def to_inner_iid(self, raw_iid):
        """
        Convert an item raw id to an inner id.
        :param raw_iid:
        :return:
        """
        try:
            return self._raw2inner_id_items[raw_iid]
        except KeyError:
            raise ValueError(
                f"item {str(raw_iid)} is not part of the trainset."
            )

    def to_raw_iid(self, iiid):
        """
        Convert an irtem inner id to a raw id
        :param iiid:
        :return:
        """
        if self._raw2inner_id_items is None:
            self._raw2inner_id_items = {
                inner: raw for (raw, inner) in
                self._raw2inner_id_items.items()
            }
        try:
            return self._raw2inner_id_items[iiid]
        except KeyError:
            raise ValueError(
                f"{str(iiid)} is not a valid inner id"
            )

    def knows_user(self, uid):
        """
        Indicate if the user is part of the trainset.
        :param uid:
        :return:
        """
        return uid in self.ur

    def knows_item(self, iid):
        """
        Indicate if the item is part of the trainset.
        :param iid:
        :return:
        """
        return iid in self.ir

    def build_testset(self):
        """
        :return: A list of ratings that can be used as a testset in the :meth: test()
        """
        return [
            (self.to_raw_uid(u), self.to_raw_iid(i), r)
            for (u, i, r) in self.all_ratings()
        ]

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