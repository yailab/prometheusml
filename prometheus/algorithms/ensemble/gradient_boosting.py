# keep the seed constant -- not sure it's required for the function #TODO check with Yannis
from random import seed

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from prometheus.algorithms.utils import compute_class
from prometheus.data_processing.scale_data import select_scaler
from prometheus.model_selection.compare_to_scikit_default import (
    compare_to_scikit_default,
)

seed(42)
np.random.seed(42)

__all__ = ["gradient_boosting_model", "gradient_boosting_model_inference"]

# name for predictive variable
predictive_variable_name = "target"


def _parameter_input(X_train, model_type="classifier", **kwargs):
    """Function that specifies parameters for the Random Forest Regressor and Classifier

    The parameter specification is based on the paper:
    Hyperparameters and Tuning Strategies for Random Forest
    by Philipp Probst, Marvin Wright and Anne-Laure Boulesteix, February 27, 2019
    https://arxiv.org/pdf/1804.03515.pdf

    :param X_train:  X input data
    :param automatic: are the parameters specified manually (by user) or use automatic param
    :return: parameters for regressor
             parameters for classifier
    """
    # regressor parameters
    mtyr = len(list(X_train))  # no. of selected variables according to the above paper

    if model_type == "regressor":
        param = {
            "n_estimators": [50, 100, 200, 300, 500],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "subsample": np.linspace(0.1, 1, 60),
            "validation_fraction": np.linspace(0.1, 0.5, 10),
            "max_depth": sp_randint(3, 15),
            "max_features": sp_randint(max(int(mtyr / 6), 1), max(int(mtyr / 4), 1)),
            "min_samples_split": [2, 3, 5, 7, 9],
            "criterion": ["mse"],
        }

        param.update(kwargs)

    # classifier parameters
    elif model_type == "classifier":
        param = {
            "n_estimators": [50, 100, 200, 300, 500],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "subsample": np.linspace(0.1, 1, 60),
            "max_depth": sp_randint(3, 15),
            "max_features": sp_randint(
                max(int(np.sqrt(mtyr) - 1), 1), max(int(np.sqrt(mtyr) + 1), 1)
            ),
            "min_samples_split": [2, 3, 5, 7, 9],
        }
        param.update(kwargs)

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    return param


def gradient_boosting_model(
    X_train,
    y_train,
    groups=None,
    scale_data: bool = False,
    cv_strategy="simple",
    tuning_strategy="RGS",
    model_type="classifier",
    **kwargs,
):
    """
    Function to tune the model

    :param X_train: input
    :param y_train: output
    :param scale_data: boolean or string
    :param tuning_strategy: RGS - random grid searcg, GS - grid search, etc
    :param type: # model type i.e. regressor of classifier
    :return:
    """
    # split the param dict
    algo_kwargs = kwargs.get("algo_params")
    hyper_kwargs = kwargs.get("hyper_params")

    # define scaler - Note: RF does not require scaling per se, however scaling might affect model performance
    if not scale_data:
        scaler = None
    elif scale_data:
        scaler = select_scaler()
    else:
        scaler = select_scaler(scaler_type=scale_data)

    # select the algorithm - raise errors for any other algorithm for now
    if model_type == "regressor":
        algorithm = GradientBoostingRegressor()
        param = _parameter_input(X_train)

    elif model_type == "classifier":
        algorithm = GradientBoostingClassifier()
        param = _parameter_input(X_train, model_type="classifier", **algo_kwargs)

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    if model_type == "regressor":
        # tune the regressor
        if cv_strategy == "simple":
            cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            print("Doing Repeated-KFold-CV for regression")

        elif cv_strategy == "complex":
            no_of_splits = len(
                np.unique(groups)
            )  # number of slits is equal to the number of groups
            cv = StratifiedGroupKFold(n_splits=no_of_splits)
            print("Doing Group-CV for regression")

        else:
            raise ValueError("no CV strategy selected for regression")

    elif model_type == "classifier":
        # tune the classifier
        # note: the classifier is using the normalized gini

        # use a group K-fold method to simulate deployment on different cells
        if cv_strategy == "simple":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            print("Doing Stratified-CV for classification")

        elif cv_strategy == "complex":
            if groups is None:
                # Stratified k-fold for obtaining balanced training datasets
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
                print("Doing Repeated Stratified-CV for classification")

            else:
                no_of_splits = len(
                    np.unique(groups)
                )  # number of slits is equal to the number of groups
                cv = StratifiedGroupKFold(n_splits=no_of_splits)
                print("Doing Group-CV for classification")

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    # select the tuning strategy
    if tuning_strategy == "RGS":  # Randomised Grid-search with CV
        search_strategy = RandomizedSearchCV(
            algorithm, param_distributions=param, cv=cv, verbose=2, **hyper_kwargs
        )
        pipe = Pipeline([("scaler", scaler), ("search_strategy", search_strategy)])

        # fit model according to the method of cv used
        if groups is None:
            pipe.fit(X_train, y_train)

        else:
            pipe.fit(X_train, y_train, group=groups)

    else:
        raise ValueError("Search strategy not supported!")

    print(f"The pipeline is: {pipe.named_steps}")

    # compare tuned model to the default scikit values
    pipeline_default = Pipeline(steps=[("scaler", scaler), ("algorithm", algorithm)])
    model, model_default = compare_to_scikit_default(
        pipe, pipeline_default, X_train, y_train, model_type
    )

    return model, model_default


def gradient_boosting_model_inference(
    X, model, decision_threshold=None, predict_proba=True, model_type="classifier"
):

    if model_type == "classifier":
        if predict_proba:
            y_prob = model.predict_proba(X)  # take only the probability of the event

            # compute the class based on the decision threshold decided during the cross-validation
            y = compute_class(y_prob, decision_threshold=decision_threshold)
            # compute the confidence interval
            CI = np.random.choice(
                np.arange(1, 12, 0.5), len(y_prob)
            )  # TODO: compute confidence interval using Jacks Knife
        else:
            raise ValueError("Hard class prediction not implemented")

        # calculate the event proba
        event_proba = np.round(y_prob.flatten()[1::2] * 100, 2)

        results_dictionary = {
            "prediction_class": y,
            "prediction_event_prob": event_proba,
            "prediction_CI": CI,
            "decision_threshold": decision_threshold,
        }
    elif model_type == "regressor":
        y = model.predict(X)  # take only the probability of the event
        CI = np.random.choice(
            np.arange(1, 12, 0.5), len(y)
        )  # TODO: compute confidence interval using Jacks Knife

        results_dictionary = {
            "prediction": y,
            "prediction_CI": CI,
        }
    else:
        raise ValueError(f"Model type {model_type} not currently supported!")

    return results_dictionary
