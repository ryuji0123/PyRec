from collections import namedtuple

class PredictionImpossible(Exception):
    """
    Exception raised when a prdiction is impossible.
    """
    pass

class Prediction(namedtuple(
    "Prediction", ["uid", "iid", "r_ui", "est", "details"]
)):
    """
    A named tuple for storing the results of a prediction.
    """
    __slots__ = ()

    def __str__(self):
        return ""
