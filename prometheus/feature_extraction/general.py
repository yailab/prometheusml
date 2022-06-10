""" General feature transformations. """

from typing import Optional

import numpy as np
import pandas as pd

from prometheus.data_processing.basic_data_checks import basic_data_processing


def data_feature_engineering(
    data: pd.DataFrame, transforms: list, use_type: Optional[str] = "training"
) -> pd.DataFrame:

    # TODO: support also categorical data
    # subset only the numerical data
    df_numerics_only = data.select_dtypes(include=np.number)

    # if no transformations selected then return the input data
    if not transforms:
        out_data = df_numerics_only
    else:
        pass

    # basic data processing and cleaning
    data_out, _ = basic_data_processing(out_data)

    return data_out
