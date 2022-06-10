from random import seed

import numpy as np
from sklearn.linear_model import LogisticRegression

# Stratified kFold
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

# import from utils
from prometheus.algorithms.utils import compute_class, compute_dataset_imbalance
from prometheus.data_processing.scale_data import select_scaler
from prometheus.model_selection.compare_to_scikit_default import (
    compare_to_scikit_default,
)

seed(42)
np.random.seed(42)


__all__ = ["logistic_regression_model", "logistic_regression_model_inference"]


def _parameter_input(model_type="classifier", **kwargs):
    """Function that specifies parameters for the Logistic Regression Classifier

    :param automatic: are the parameters specified manually (by user) or use automatic param
    :return: parameters for regressor
             parameters for classifier
    """
    # regressor parameters
    if model_type == "regressor":
        raise ValueError("Regressor not supported by Logistic Regression!")

    elif model_type == "classifier":

        param = {
            "penalty": ["l1", "l2"],
            "C": np.arange(0.05, 10, 0.05),
        }

        param.update(kwargs)

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    return param


def logistic_regression_model(
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
        raise ValueError("Regressor not supported by Logistic Regression!")
    elif model_type == "classifier":
        balanced_weights = compute_dataset_imbalance(X_train, y_train)
        algorithm = LogisticRegression(
            max_iter=500,
            multi_class="ovr",
            solver="liblinear",
            class_weight=balanced_weights,
            random_state=1,
        )
        param = _parameter_input(model_type="classifier", **algo_kwargs)

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    if model_type == "classifier":
        # define scoring metric - classifier
        scoring_metric = "neg_log_loss"
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


def logistic_regression_model_inference(
    X, model, decision_threshold, predict_proba=True, model_type="classifier"
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

        # calculate the event probability
        event_proba = np.round(y_prob.flatten()[1::2] * 100, 2)

        results_dictionary = {
            "prediction_class": y,
            "prediction_event_prob": event_proba,
            "prediction_CI": CI,
            "decision_threshold": decision_threshold,
        }

    else:
        raise ValueError(f"Model type {model_type} not currently supported!")

    return results_dictionary
