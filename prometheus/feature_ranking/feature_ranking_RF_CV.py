"""
Functions to create feature ranking using Random Forest and cross-validation
"""

from random import seed
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

from prometheus.algorithms.ensemble.random_forest import RF_model
from prometheus.algorithms.utils import create_input_output

NDArrayFloat = npt.NDArray[np.float_]

# TODO: centralised way to control seeds
seed(42)
np.random.seed(42)


# name for predictive variable
predictive_variable_name = "target"

__all__ = ["feature_selection_with_RF_and_CV"]


def _plot_feature_ranking(
    f_importance_dataframe: pd.DataFrame,
    max_features_display: int = 25,
    report_type: str = "platform",
) -> plt.figure:
    """
    :param f_importance_dataframe: pandas dataframe with the feature name and importance in decreasing order
    from most important to lowest
    :param max_features_display: maximum no. of features to display after which a sum of all other feature
    contribution is shown
    :param report_type: weather the figure is for the platform or for the downloadable report
    :return: figure with feature importance as bar plot
    """

    # specify colours for the plot
    n = (
        max_features_display + 1
    )  # number of colors based on the maximum features to display plus one (the sum contribution of all other features)
    color = plt.cm.Wistia(np.linspace(0, 1, n))

    if report_type == "platform":
        color_annotate = "w"
    elif report_type == "client":
        color_annotate = "k"
    else:
        raise ValueError("Report type not selected!")
    font_size = 30
    axis_font_size = 25

    # only display up to 'max_features_display' and add feature name 'Others' and take the sum of contributions
    features_name_all = [
        str(idx + 1) + ". " + f
        for idx, f in enumerate(f_importance_dataframe.feature_name)
    ]
    features_name = features_name_all[:max_features_display] + ["Other"]
    features_importance = list(
        f_importance_dataframe.feature_importance[:max_features_display]
    ) + [abs(np.mean(f_importance_dataframe.feature_importance[max_features_display:]))]

    # inverse the order of the features such the highest value is at the top
    features_name = features_name[::-1]
    features_importance = features_importance[::-1]

    fig = plt.figure(figsize=(20, 20))
    plt.xlabel("Feature importance", fontsize=font_size, color=color_annotate)

    # plot reversed list to see the highest contributing feature first
    plt.barh(features_name, features_importance, color=color, align="center")

    plt.yticks(rotation=0, fontsize=font_size, color=color_annotate)
    plt.xticks(fontsize=axis_font_size, color=color_annotate)

    plt.tight_layout()
    plt.grid(which="major", axis="x", linestyle="--", color=color_annotate)
    plt.box(False)

    return fig


def _create_tune_model(
    X: NDArrayFloat,
    y: NDArrayFloat,
    group: Optional[Any] = None,
    alg_type: str = "classifier",
):
    print(f"------> Prometheus is using random forest {alg_type} to rank you features")
    # define a hyperparameter for random search
    params_dict = {"algo_params": {}, "hyper_params": {"n_iter": 15}}

    model, _ = RF_model(
        X, y, groups=group, tuning_strategy="RGS", model_type=alg_type, **params_dict
    )

    return model


def _split_stratified_into_train_val_test(
    df_input: pd.DataFrame,
    stratify_colname: str = predictive_variable_name,
    model_type: str = "classifier",
    frac_train: float = 0.33,
    frac_val: float = 0.33,
    frac_test: float = 0.34,
    random_state: Optional[int] = None,
):
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    source: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            "fractions %f, %f, %f do not add up to 1.0"
            % (frac_train, frac_val, frac_test)
        )

    if stratify_colname not in df_input.columns:
        raise ValueError("%s is not a column in the dataframe" % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[
        [stratify_colname]
    ]  # Dataframe of just the column on which to stratify.

    # stratify only is this is a classifier otherwise set stratification to None
    if model_type == "classifier":
        print(
            "------> Prometheus is splitting data for ranking features as a classifier!"
        )
        stratify_y = df_input[
            [stratify_colname]
        ]  # Dataframe of just the column on which to stratify.
    elif model_type == "regressor":
        print(
            "------> Prometheus is splitting data for ranking features as a regressor!"
        )
        stratify_y = None
        stratify_y_temp = None
    else:
        raise ValueError("Model type not selected!")

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X,
        y,
        stratify=stratify_y,
        test_size=(1.0 - frac_train),
        random_state=random_state,
    )

    # second stratification pass
    if model_type == "classifier":
        stratify_y_temp = y_temp  # Dataframe of just the column on which to stratify.
    else:
        pass
    # Split original dataframe into train and temp dataframes.

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=stratify_y_temp,
        test_size=relative_frac_test,
        random_state=random_state,
    )

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def feature_selection_with_RF_and_CV(
    data: pd.DataFrame, display_features: int = 25, alg_type: str = "classifier"
) -> Tuple[pd.DataFrame, plt.figure, plt.figure]:
    """
    Function that returns feature importance of the model based on Random Forest cross-validation:
        1. Use stratified kfold to partition the feature selection dataset in 3 (balanced) folds
        2. Train three models on each fold and calculate the feature importance for each of the feature
        3. Aggregate the results of each feature importance for each model and compute the final feature importance as a mean

    :param alg_type:
    :param data:
    :param display_features:
    :return:
    """
    print(f"-----> NaNs in data for feature ranking: {data.isna().sum().sum()}")

    # create 3 data folds of equal size and balanced

    min_no_of_samples = 150  # minimum number of samples needed for feature ranking

    if alg_type == "classifier":

        # check number of classes
        if len(data.groupby("target").size()) > 2:
            raise ValueError("Multi class not supported!")
        else:
            # number of samples in each class
            class_0 = data.groupby("target").size()[0]
            class_1 = data.groupby("target").size()[1]

            if class_0 < int(min_no_of_samples / 3) or class_1 < int(
                min_no_of_samples / 3
            ):
                samples = int(min(class_0, class_1) / 2)
                # toggle replacement based on number of samples
                if samples < 10:
                    samples = min(class_0, class_1)
                    replace = True
                else:
                    replace = False

            else:
                samples = 50  # constrained to 50 samples due to computational demand
                replace = False

            d_1 = data.groupby("target").sample(
                n=samples, replace=replace, random_state=1
            )
            d_2 = data.groupby("target").sample(
                n=samples, replace=replace, random_state=2
            )
            d_3 = data.groupby("target").sample(
                n=samples, replace=replace, random_state=3
            )

    elif alg_type == "regressor":

        if len(data) < min_no_of_samples:
            samples = int(len(data) * 0.5)
            replace = True
        else:
            samples = 50  # constrained to 50 samples due to computational demand
            replace = False

        d_1 = data.sample(n=samples, replace=replace, random_state=1)
        d_2 = data.sample(n=samples, replace=replace, random_state=2)
        d_3 = data.sample(n=samples, replace=replace, random_state=3)

    else:
        raise ValueError(f"Model type not supported: {alg_type}!")

    # TODO automate this for any no of models (use dictionaries)
    # tune model and fit it on each dataset

    X_1, y_1, group1 = create_input_output(
        d_1, model_type=alg_type, output=predictive_variable_name
    )
    X_2, y_2, group2 = create_input_output(
        d_2, model_type=alg_type, output=predictive_variable_name
    )
    X_3, y_3, group3 = create_input_output(
        d_3, model_type=alg_type, output=predictive_variable_name
    )

    model_1 = _create_tune_model(X_1, y_1, alg_type=alg_type)
    model_2 = _create_tune_model(X_2, y_2, alg_type=alg_type)
    model_3 = _create_tune_model(X_3, y_3, alg_type=alg_type)

    # calculate random forest feature importance
    imp_m_1 = model_1.named_steps.get("algorithm").feature_importances_
    imp_m_2 = model_2.named_steps.get("algorithm").feature_importances_
    imp_m_3 = model_3.named_steps.get("algorithm").feature_importances_

    importances = np.mean(
        [imp_m_1, imp_m_2, imp_m_3], axis=0
    )  # calculate the average per vector
    indices = np.argsort(importances)[0:]

    # create a dataframe with the extracted feature importance
    f_importance_dataframe = pd.DataFrame()
    f_importance_dataframe["feature_name"] = X_1.columns[indices]
    f_importance_dataframe["feature_importance"] = importances[indices]
    f_importance_dataframe.sort_values(
        ["feature_importance"], ascending=False, axis=0, inplace=True
    )

    # TODO: move the plots out of this function
    figure_platform = _plot_feature_ranking(
        f_importance_dataframe, max_features_display=display_features
    )
    figure_report = _plot_feature_ranking(
        f_importance_dataframe,
        max_features_display=display_features,
        report_type="client",
    )

    return f_importance_dataframe, figure_platform, figure_report
