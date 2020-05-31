from .predictions import PredictionImpossible, Prediction

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

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """
        Compute the rating prediction for given user and item
        :param uid:
        :param iid:
        :param r_ui:
        :param clip:
        :param verbose:
        :return:
        """
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            # unknown
            iuid = "UKN__" + str(uid)

        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = "UKN__" + str(iid)

        details = {}

        try:
            est = self.estimate(iuid, iiid)
        except PredictionImpossible as e:
            est = self.default_prediction()
            details["was_impossible"] = True
            details["reason"] = str(e)

        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid=uid, iid=iid, r_ui=r_ui, est=est, details=details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        """
        used when the PredictionImpossible exception us raised during a call to predict()
        :return:
        """
        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        """
        test the algorithm on given testset.
        :param testset:
        :param verbose:
        :return:
        """
        return [
            self.predict(
                uid=uid, iid=iid, r_ui=r_ui_trans, verbose=verbose
            ) for (uid, iid, r_ui_trans) in testset
        ]