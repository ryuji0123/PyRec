cimport numpy as np
import numpy as np

from .algo_base import AlgoBase
from ..utils import get_rng

class SVD(AlgoBase):
    """
    The famous *SVD* algorithms.
    :param n_factors: The number of factors.
    :param n_epochs: The number of iteration of SGD procedure.
    :param biased(bool):
    :param init_mean: The mean of the normal distiribution for factor vectors initialization.
    :param init_std_dev: The standard deviation of the normal distribution for factor vectors initialization.
    :param random_state (int, RandomState instance, None): Determines the RNG that will be used for initialization.
    :param verbose: If True, prints the current epoch.

    :attribute pu: The user factors
    :attribute qi: The item factors
    :attribute bu: The user biases
    :attribute bi: The item biases
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
        """
        Following algorithms are based on BelKor solution
        (https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf)
        :param trainset:
        :return:
        """
        #user biases
        cdef np.ndarray[np.double_t] bu

        #item biases
        cdef np.ndarray[np.double_t] bi

        #user factors
        cdef np.ndarray[np.double_t, ndim=2] pu

        #item factors
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i, f
        cdef double r, err, dot

        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double global_mean = self.trainset.global_mean

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        # random generator
        rng = get_rng(self.random_state)

        pu = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for cur_epoch in range(self.n_epochs):
            if self.verbose:
                print(f"Processing epoch {cur_epoch}")
            for u, i, r in trainset.all_ratings():
                dot = 0
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
            err = r - (global_mean + bu[u] + bi[i] + dot)

            #update biases
            if self.biased:
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

            # compute numerators and denominators
            for f in range(self.n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                qi[i, f] += lr_qi * (err * puf - reg_qi * qif)


        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi