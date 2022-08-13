from molmagic.split import stoichiometric_split
import numpy as np


def test_stoichiometric_split():
    X = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [3, 3, 3, 3]
    ])
    y = np.ones_like(X)[:, 0]

    X_train, X_test, y_train, y_test = stoichiometric_split(X, y)

    expected_X_train = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ])
    expected_y_train = np.ones_like(expected_X_train)[:, 0]

    expected_X_test = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ])
    expected_y_test = np.ones_like(expected_X_test)[:, 0]

    assert (X_train == expected_X_train).all()
    assert (X_test == expected_X_test).all()
    assert (y_train == expected_y_train).all()
    assert (y_test == expected_y_test).all()