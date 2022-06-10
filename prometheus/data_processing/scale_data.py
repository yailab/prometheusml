"""
Function to apply a scaler prior to training.

Notes:
- for an in-depth analysis for the effect of scaler see:
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
"""

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# TODO implement all scalers imported below


def select_scaler(scaler_type: str = "Standard"):

    if scaler_type == "Standard":
        scaler = StandardScaler()
    elif scaler_type == "MinMax":
        scaler = MinMaxScaler()
    elif scaler_type == "Robust":
        scaler = RobustScaler()
    else:
        raise ValueError(
            "Scaler not specified - select one of the supported scalers: StandardScaler, MinMaxScaler, Robust"
        )
    return scaler
