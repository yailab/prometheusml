"""A collection of helper functions for the various tasks"""
import os
import sys
from functools import partial
from typing import List, Optional

import pandas as pd
from flask import current_app

from webapp.utils import df_is_numeric_eu, df_is_numeric_us


def get_intermediate_path(user_id: int, project_id: int, use_type: str) -> str:
    root_path_app = os.path.dirname(current_app.instance_path)
    intermediate_path = os.path.join(
        root_path_app,
        current_app.config["UPLOAD_FOLDER"],
        use_type,
        str(user_id),
        str(project_id),
    )
    return intermediate_path


def get_uploaded_file_paths(
    user_id: int, project_id: int, use_type: str = ""
) -> List[str]:
    # -------------------------------------------------------------------- #
    # Get the dir of the uploaded dataset(s)
    try:
        intermediate_path = get_intermediate_path(user_id, project_id, use_type)

        # the app specific filename of the uploaded file
        custom_filename = (
            "uploaded_dataset_" + str(user_id) + "_" + str(project_id) + ".csv"
        )
        # Get only the directory names (that's where we save the uploaded data)
        dirnames = next(os.walk(intermediate_path))[1]

        # Create the list of the uploaded file paths
        uploaded_file_paths = [
            os.path.join(intermediate_path, str(dirname), custom_filename)
            for dirname in dirnames
        ]

        return uploaded_file_paths
    except OSError as e:
        current_app.logger.exception("Filesystem error: %s" % e)
        raise


def read_uploaded_files(
    uploaded_file_paths: List[str],
    predictive_variable_name: Optional[str] = None,
    use_type: Optional[str] = "training",
    use_cols: Optional[list] = None,
) -> List[pd.DataFrame]:
    try:
        if use_type == "inference":
            # Only one file uploaded in the inference case
            data_sample = pd.read_csv(
                uploaded_file_paths[0], nrows=5, usecols=use_cols
            )  # sample the data
            if df_is_numeric_eu(data_sample):
                decimal, thousands = ",", "."
            elif df_is_numeric_us(data_sample):
                decimal, thousands = ".", ","
            else:
                raise ValueError(
                    "The numerical values are of no known format or malformed!"
                )
            data_raw = pd.read_csv(
                uploaded_file_paths[0],
                decimal=decimal,
                thousands=thousands,
                usecols=use_cols,
            )
            # change the target feature name to `target`
            data_raw.rename(columns={predictive_variable_name: "target"}, inplace=True)
            data_raw_list = [data_raw]
        else:
            # sample the datasets
            read_csv_top_func = partial(pd.read_csv, nrows=5, usecols=use_cols)
            data_sample_list = list(
                map(read_csv_top_func, uploaded_file_paths)
            )  # get list of dfs

            if all([df_is_numeric_eu(df) for df in data_sample_list]):
                decimal, thousands = ",", "."
            elif all([df_is_numeric_us(df) for df in data_sample_list]):
                decimal, thousands = ".", ","
            else:
                raise ValueError(
                    "The numerical values are of no known format or malformed!"
                )

            # For training and re-training read multiple files
            read_csv_partial_locale = partial(
                pd.read_csv, usecols=use_cols, decimal=decimal, thousands=thousands
            )
            data_retrain_list_raw = list(
                map(read_csv_partial_locale, uploaded_file_paths)
            )
            # change the target feature name to `target`
            data_raw_list = [
                df.rename(columns={predictive_variable_name: "target"})
                for df in data_retrain_list_raw
            ]

        return data_raw_list
    except pd.errors.ParserError:
        # log the current exception along with the trace information, prepended with a message
        current_app.logger.exception(
            "The csv file is not formatted right... Cannot be parsed correctly... "
        )
        # Raise the error so that it can be caught by the upstream server
        raise
    except Exception:
        # pass the exception instances in the `exc_info` argument
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        raise
