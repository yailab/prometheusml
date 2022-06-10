from typing import Tuple

import numpy as np
import pandas as pd


def basic_data_processing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    https://stackoverflow.com/questions/59384802/how-can-i-automatically-detect-if-a-colum-is-categorical
    https://stackoverflow.com/questions/35826912/what-is-a-good-heuristic-to-detect-if-a-column-in-a-pandas-dataframe-is-categori
    """

    # Keep only numerical features
    data = data.select_dtypes(include=np.number)

    # Replace NaNs with median - return median and save as it needs to be used for inference
    median_values_for_each_feature = data.median()
    data.fillna(median_values_for_each_feature, inplace=True)

    # Check for inf/-inf and replace with maximum in tha columns
    max_per_feature = data.max()
    data.replace(np.inf, max_per_feature, inplace=True)

    min_per_feature = data.min()
    data.replace(-np.inf, min_per_feature, inplace=True)

    # TODO: check for outliers and very large numbers
    # TODO: check for 0 and replace with 0.00001 - does it make sense?

    return data, median_values_for_each_feature
