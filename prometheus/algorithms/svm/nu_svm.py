# keep the seed constant -- not sure it's required for the function #TODO check with Yannis
from random import seed

import numpy as np

# Stratified kFold
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC, NuSVR

# import from utils
from prometheus.algorithms.utils import compute_class, compute_dataset_imbalance
from prometheus.data_processing.scale_data import select_scaler
from prometheus.model_selection.compare_to_scikit_default import (
    compare_to_scikit_default,
)

seed(42)
np.random.seed(42)


__all__ = ["nu_svm_model", "nu_svm_model_inference"]


def _parameter_input(model_type="classifier", **kwargs):
    """Function that specifies parameters for the Random Forest Regressor and Classifier

    :param automatic: are the parameters specified manually (by user) or use automatic param
    :return: parameters for regressor
             parameters for classifier
    """
    # regressor parameters
    if model_type == "regressor":
        param = {
            "nu": np.arange(0.1, 1.0, 0.05),
            "C": np.arange(0.05, 10, 0.05),  # np.arange(0.0001, 10.0, 0.01),
            "kernel": ["linear", "poly", "rbf"],
            "degree": [1, 2, 3],
            "gamma": ["scale", "auto", 2**-15],
        }

        param.update(kwargs)

    # classifier parameters
    elif model_type == "classifier":
        param = {
            "nu": np.arange(0.1, 1.0, 0.05),
            "kernel": ["linear", "poly", "rbf"],
            "degree": [1, 2, 3],
            "gamma": ["scale", "auto", 2**-15],
        }

        param.update(kwargs)

    else:
        raise ValueError(f"Model type not supported: {model_type}!")

    return param


def nu_svm_model(
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

    :param groups:
    :param cv_strategy:
    :param X_train: input
    :param y_train: output
    :param scale_data: boolean or string
    :param automatic:
        whether the user selects automatic tuning or manual Note: for feature selection is always automatic
    :param tuning_strategy: RGS - random grid search, GS - grid search, etc
    :param model_type: # model type i.e. regressor of classifier
    :return:
    """
    # split the param dict
    algo_kwargs = kwargs.get("algo_params")
    hyper_kwargs = kwargs.get("hyper_params")

    print(f"The algorithm parameters are: \n {algo_kwargs}")
    print(f"The hyper parameters are: \n {hyper_kwargs}")

    # define scaler
    if not scale_data:
        scaler = None
    elif scale_data:
        scaler = select_scaler()
    else:
        scaler = select_scaler(scaler_type=scale_data)

    # select the algorithm - raise errors for any other algorithm for now
    if model_type == "regressor":
        algorithm = NuSVR()
        param = _parameter_input(model_type="regressor", **algo_kwargs)

    elif model_type == "classifier":

        # compute model imbalance
        balanced_weights = compute_dataset_imbalance(X_train, y_train)
        algorithm = NuSVC(probability=True, class_weight=balanced_weights)
        param = _parameter_input(model_type="classifier", **algo_kwargs)

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

    elif model_type == "classifier":
        # tune the classifier
        # note: the classifier is using the normalized gini

        # use a group K-fold method to simulate deployment on different cells
        if cv_strategy == "simple":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            print("Doing Stratified-CV")

        elif cv_strategy == "complex":
            if groups is None:
                # Stratified k-fold for obtaining balanced training datasets
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
                print("Doing Repeated Stratified-CV")

            else:
                no_of_splits = len(
                    np.unique(groups)
                )  # number of slits is equal to the number of groups
                cv = StratifiedGroupKFold(n_splits=no_of_splits)
                print("Doing Group-CV")

        # define scoring metric
        scoring_metric = "neg_log_loss"  # _gini_normalized
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


def nu_svm_model_inference(
    X, model, decision_threshold=None, predict_proba=True, model_type="classifier"
):

    if model_type == "classifier":
        if predict_proba:
            y_prob = model.predict_proba(X)  # take only the probability of the event

            # compute the class based on the decision threshold decided during the cross-calidation
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
        CI = None  # TODO: compute confidence interval using Jacks Knife

        results_dictionary = {
            "prediction": y,
            "prediction_CI": CI,
        }
    else:
        raise ValueError(f"Model type {model_type} not currently supported!")

    return results_dictionary
