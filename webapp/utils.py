import base64
import io
import json

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
from PIL import Image


def get_encoded_img(image_path: str):
    img = Image.open(image_path, mode="r")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="SVG")
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode("ascii")
    return my_encoded_img


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)


def _is_numeric_us(array: pd.Series) -> bool:
    """Check if a pandas series is numeric in US style. Comma (,) separated thousands
    and point (.) decimal separator.
    """
    us_num_regex_pattern = "^(\\d*\\.?\\d+|\\d{1,3}(,\\d{3})*(\\.\\d+)?)$"
    if pd.api.types.is_numeric_dtype(array):
        rtn = True
    elif (
        pd_types.is_object_dtype(array) and array.str.match(us_num_regex_pattern).all()
    ):
        rtn = True
    else:
        rtn = False

    return rtn


def df_is_numeric_us(data: pd.DataFrame) -> bool:
    """Check if a dataframe follows the US numeric style."""
    if data.apply(_is_numeric_us).all():
        rtn = True
    else:
        rtn = False

    return rtn


def _is_numeric_eu(array: pd.Series) -> bool:
    """Check if a pandas series is numeric in EU style. Point (.) separated thousands
    and comma (,) decimal separator.
    """
    eu_num_regex_pattern = (
        "(?=^[^,]*,[^,]*$)(?=^(\\d*\\,?\\d+|\\d{1,3}(.\\d{3})*(\\,\\d+)?)$)"
    )
    if pd_types.is_object_dtype(array) and array.str.match(eu_num_regex_pattern).all():
        rtn = True
    else:
        rtn = False

    return rtn


def df_is_numeric_eu(data: pd.DataFrame) -> bool:
    """Check if a dataframe follows the EU numeric style."""
    # Due to regex implementation we check if data is a mixture of int and eu style
    num_eu_mask = data.apply(_is_numeric_eu)
    int_mask = data.apply(pd_types.is_signed_integer_dtype)
    eu_int_mask = num_eu_mask | int_mask

    if eu_int_mask.all():
        rtn = True
    else:
        rtn = False

    return rtn


def is_numeric_data(array: pd.Series) -> bool:
    """Check if a numpy array is numeric"""
    if _is_numeric_eu(array) or _is_numeric_us(array):
        rtn = True
    else:
        rtn = False

    return rtn
