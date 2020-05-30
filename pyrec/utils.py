import numpy as np
import numbers

def get_rng(random_state):
    """
    Return a validated RNG
    :param random_state:
    :return:
    """
    if random_state is None:
        return np.random.RandomState(0)

    if isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)

    if isinstance(random_state, np.random.RandomState):
        return random_state

    raise ValueError(
        f"Wrong random state. Expecting None, an int or a numpy RandonState instance, got a {type(random_state)}"
    )