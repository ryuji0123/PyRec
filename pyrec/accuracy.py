
import numpy as np


def rmse(predictions, verbose=False):
    """
    Compute Root Mean Squared Error.
    :param predictions:
    :param verbose:
    :return:
    """
    if not predictions:
        raise ValueError("Prediction list is empty.")

    rmse = np.sqrt(
        np.mean([
            float((true_r - est)**2)
            for (_, _, true_r, est, _) in predictions
        ])
    )

    if verbose:
        print(f"RMSE: {rmse}")
    return rmse

def mae(predictions, verbose=False):
    """
    Compute Mean Absolute Error.
    :param predictions:
    :param verbose:
    :return:
    """
    if not predictions:
        raise ValueError("Prediction list is empty.")

    mae = np.mean([
            float(abs(true_r - est)**2)
            for (_, _, true_r, est, _) in predictions
        ])

    if verbose:
        print(f"MAE: {mae}")
    return mae
