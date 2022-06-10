"""Metrics for algorithms"""
from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float_]


def root_mean_abs_percent_error(
    y_true: NDArrayFloat, y_pred: NDArrayFloat
) -> Union[Any, np.ndarray]:
    """
    Compute Root Mean Squared Percentage Error between two arrays.
    """
    return np.round(np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)))) * 100, 2)


def mean_abs_percent_error(
    y_true: NDArrayFloat, y_pred: NDArrayFloat
) -> Union[Any, np.ndarray]:
    """
    Computes Mean Absolute Percentage Error between two arrays. Only for regressors.
    """
    mape = np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)  # unit [%]
    return mape


def PEP_PLP(
    y_r: NDArrayFloat, y_r_hat: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Function to compute the Percent of Early Predictions (PEP) and Percent of Later Predictions (PLP).
    """
    pep_vector = []  # percent of early predictions
    plp_vector = []  # percent of late predictions
    for idx_i, i in enumerate(y_r):
        if y_r_hat[idx_i] < i:
            pep_vector.append(i)
        else:
            plp_vector.append(i)
    pep = np.round(len(pep_vector) / len(y_r) * 100, 2)
    plp = np.round(len(plp_vector) / len(y_r) * 100, 2)

    return pep, plp
