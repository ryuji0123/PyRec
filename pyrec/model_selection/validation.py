import time

from joblib import Parallel, delayed

from .split import get_cv


def cross_validate(algo, data, measures=["rmse", "mae"], cv=None, return_train_measuers=False, n_jobs=1, pre_dispatch="2*n_jobs", verbose=False):
    """
    Run a cross validation procedure for a given algorithm, reporting accuracy measures and computation times.
    :param algo:
    :param data:
    :param measures:
    :param cv:
    :param return_train_measuers:
    :param n_jobs:
    :param pre_dispatch:
    :param verbose:
    :return:
    """

    measures = [m.lower() for m in measures]

    cv = get_cv(cv)
    delayed_list = (
        delayed(fit_and_score)(algo, trainset, testset, measures, return_train_measuers)
        for (trainset, testset) in cv.split(data)
    )
    out = Parallel(
        n_jobs=n_jobs, pre_dispatch=pre_dispatch
    )(delayed_list)
    return

def fit_and_score(algo, trainset, testset, measures, return_train_measures=False):
    """
    Helper method that trains an algorithm and compute accuracy measures on a testset.
    Also report train and test times.
    :param algo:
    :param trainset:
    :param testset:
    :param measures:
    :param return_train_measures:
    :return:
    """
    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test
    print(f"fit: {fit_time}, test: {test_time}")

    if return_train_measures:
        train_predictions = algo.test(trainset.build_testset())
