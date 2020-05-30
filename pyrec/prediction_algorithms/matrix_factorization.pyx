
from .algo_base import AlgoBase

class SVD(AlgoBase):
    """
    The famous *SVD* algorithms.
    """
    def __init__(
            self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
            init_std_dev=.1, lr_all=.005, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
            lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
            random_state=None, verbose=False
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)


    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):
        return

