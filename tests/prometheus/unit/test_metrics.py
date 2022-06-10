"""Module of unit tests for the metrics"""
from prometheus.metrics import mean_abs_percent_error, root_mean_abs_percent_error


def test_rounded_rmape(arr_one, arr_two):
    """
    GIVEN two arrays
    WHEN a root mean absolute percent error is computed
    THEN check the computed value is the expected
    """
    assert root_mean_abs_percent_error(arr_one, arr_two) == 1703.42


def test_rounded_mape(arr_one, arr_two):
    """
    GIVEN two arrays
    WHEN a root mean absolute percent error is computed
    THEN check the computed value is the expected
    """
    assert mean_abs_percent_error(arr_one, arr_two) == 350.84
