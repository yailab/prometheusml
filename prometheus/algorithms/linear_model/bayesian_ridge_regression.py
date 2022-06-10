# keep the seed constant -- not sure it's required for the function #TODO check with Yannis
from random import seed

import numpy as np
from sklearn.linear_model import BayesianRidge

# Stratified kFold
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline

# import from utils
from prometheus.data_processing.scale_data import select_scaler
from prometheus.model_selection.compare_to_scikit_default import (
    compare_to_scikit_default,
)

seed(42)
np.random.seed(42)


__all__ = ["bayesian_ridge_model", "bayesian_ridge_model_inference"]


def _parameter_input(model_type="classifier", **kwargs):
    """Function that specifies parameters for the Random Forest Regressor and Classifier

    :param automatic: are the parameters specified manually (by user) or use automatic param
    :return: parameters for regressor
             parameters for classifier
    """
    # regressor parameters
    if model_type == "regressor":

        param = {  # 10 ** np.arange(10) * 1e-9 # TODO add the range to the below params
            "alpha_1": [
                1e-9,
                0.5e-9,
                1e-8,
                5e-8,
                1e-7,
                0.5e-7,
                1e-6,
                0.5e-6,
                1e-5,
                0.5e-5,
                1e-4,
                0.5e-4,
                1e-3,
                0.5e-3,
            ],
            "alpha_2": [
                1e-9,
                0.5e-9,
                1e-8,
                5e-8,
                1e-7,
                0.5e-7,
                1e-6,
                0.5e-6,
                1e-5,
                0.5e-5,
                1e-4,
                0.5e-4,
                1e-3,
                0.5e-3,
            ],
            "lambda_1": [
                1e-9,
                0.5e-9,
                1e-8,
                5e-8,
                1e-7,
                0.5e-7,
                1e-6,
                0.5e-6,
                1e-5,
                0.5e-5,
                1e-4,
                0.5e-4,
                1e-3,
                0.5e-3,
            ],
            "lambda_2": [
                1e-9,
                0.5e-9,
                1e-8,
                5e-8,
                1e-7,
                0.5e-7,
                1e-6,
                0.5e-6,
                1e-5,
                0.5e-5,
                1e-4,
                0.5e-4,
                1e-3,
                0.5e-3,
            ],
        }

        param.update(kwargs)

    elif model_type == "classifier":
        raise ValueError("Classifier not supported by Bayesian Ridge!")
    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    return param


def bayesian_ridge_model(
    X_train,
    y_train,
    groups=None,
    scale_data: bool = True,
    cv_strategy="simple",
    tuning_strategy="RGS",
    model_type="classifier",
    **kwargs,
):
    """
    Function to tune the model

    :param cv_strategy:
    :param groups:
    :param X_train: input
    :param y_train: output
    :param scale_data: boolean or string
    :param tuning_strategy: RGS - random grid search, GS - grid search, etc
    :param model_type: # model type i.e. regressor of classifier
    :return:
    """
    # split the param dict
    algo_kwargs = kwargs.get("algo_params")
    hyper_kwargs = kwargs.get("hyper_params")

    # define scaler
    if not scale_data:
        scaler = None
    elif scale_data:
        scaler = select_scaler()
    else:
        scaler = select_scaler(scaler_type=scale_data)

    # select the algorithm - raise errors for any other algorithm for now
    if model_type == "regressor":
        algorithm = BayesianRidge(n_iter=500)
        param = _parameter_input(model_type="regressor", **algo_kwargs)

    elif model_type == "classifier":
        raise ValueError("Classifier not supported by Bayesian Ridge!")

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    if model_type == "regressor":
        # tune the regressor
        if cv_strategy == "simple":
            cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            print("Doing Repeated-KFold-CV")

        elif cv_strategy == "complex":
            no_of_splits = len(
                np.unique(groups)
            )  # number of slits is equal to the number of groups
            cv = StratifiedGroupKFold(n_splits=no_of_splits)
            print("Doing Group-CV")

        else:
            raise ValueError("no CV strategy selected")

        # define scoring metric
        scoring_metric = "r2"  # 'neg_mean_absolute_percentage_error'
    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    # select the tuning strategy
    if tuning_strategy == "RGS":  # Randomised Grid-search with CV
        search_strategy = RandomizedSearchCV(
            algorithm,
            param_distributions=param,
            cv=cv,
            scoring=scoring_metric,
            verbose=2,
            **hyper_kwargs,
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
    pipeline_for_default = Pipeline(
        steps=[("scaler", scaler), ("algorithm", algorithm)]
    )
    model, model_default = compare_to_scikit_default(
        pipe, pipeline_for_default, X_train, y_train, model_type
    )

    return model, model_default


def bayesian_ridge_model_inference(X, model, model_type="regressor"):

    if model_type == "classifier":
        raise ValueError("Classifier not supported by Bayesian Ridge!")

    if model_type == "regressor":
        y, sigma = model.predict(
            X, return_std=True
        )  # take only the probability of the event
        z = 2  # 90% confidence interval
        CI = z * sigma

        results_dictionary = {
            "prediction": y,
            "prediction_CI": CI,
        }
    else:
        raise ValueError(f"Model type {model_type} not currently supported!")

    return results_dictionary
