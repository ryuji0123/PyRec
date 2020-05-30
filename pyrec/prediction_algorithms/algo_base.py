
class AlgoBase(object):
    """
    Abstract class where is defined the basic behavior of a prediction algorithm.
    """

    def __init__(self, **kwargs):
        self.baseline_opotions = kwargs.get("baseline_options", {})
        self.similarity_options = kwargs.get("similarity_options", {})
        if "user_based" not in self.similarity_options:
            self.similarity_options["user_based"] = True

    def fit(self, trainset):
        """
        Train an algorithm on a given training set.
        :param trainset:
        :return:
        """
        self.trainset = trainset
        self.bu = self.bi = None

        return self