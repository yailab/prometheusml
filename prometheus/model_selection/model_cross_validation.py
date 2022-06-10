"""Collection of functions for model validation in both regression and classification"""
import numpy as np

# use balanced accuracy score
from sklearn.metrics import balanced_accuracy_score as accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from prometheus.algorithms.utils import compute_class, create_input_output
from prometheus.metrics import (
    PEP_PLP,
    mean_abs_percent_error,
    root_mean_abs_percent_error,
)

# name for predictive variable
predictive_variable_name = "target"


def cross_validate_the_models(data, model, model_type="classifier", **kwargs):
    """Function to cross-validate the models
    :param model_type:
    :param data:
    :param model:
    :return:
    """

    X_train, y_train, group = create_input_output(data, model_type=model_type)

    # Evaluate the model using a K-fold cross-validation i.e. predicts in unseen test data

    if model_type == "classifier":
        print("-----> Prometheus is doing cross-validation for classifier.")

        # TODO hard-code stratified k-fold for the classifier - it does not work with other CV strategies
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        y_hat_proba = cross_val_predict(
            model, X_train, y_train, cv=cv, method="predict_proba"
        )

        decisio_th = kwargs.get("decision_threshold")
        y_hat = compute_class(y_hat_proba, decision_threshold=decisio_th)

        return y_hat, y_hat_proba

    elif model_type == "regressor":
        print("-----> Prometheus is doing cross-validation for regressor.")

        # TODO hard-code stratified k-fold for the classifier - also add groupCV here
        y_hat = cross_val_predict(model, X_train, y_train, cv=3)
        return y_hat, None  # TODO any other option apart from returning a None?

    else:
        raise ValueError("No algorithm to validate check model type")


def calculate_mean_model_probability(y_hat_proba):
    """
    Function to calculate the model average probability per class of a classifier.
    """

    prob = []  # list to store the probability for each class

    for i in y_hat_proba.flatten()[
        1::2
    ]:  # select just the first class and calculate probability by subtracting

        if (
            np.round(i) == 0
        ):  # calculate the probability of class 0 from the output probability of class 1
            prob.append(1 - i)
        else:
            prob.append(i)

    return np.round(np.mean(prob), 3) * 100  # return the mean probability


def calculate_model_performance(
    y, y_hat_proba, model_type="classifier", decision_threshold=None
):
    """Function to calculate the performance of the classifier and regressor."""
    # TODO y_hat_proba is not a relevant name for regression
    if model_type == "classifier":

        y_hat = compute_class(y_hat_proba, decision_threshold=decision_threshold)

        precision = np.round(precision_score(y, y_hat, labels=[1, 0]) * 100, 2)
        recall = np.round(recall_score(y, y_hat, labels=[1, 0]) * 100, 2)

        f1 = np.round(f1_score(y, y_hat) * 100, 2)
        accuracy = np.round(accuracy_score(y, y_hat) * 100, 2)
        mean_class_predicted_probability = np.round(
            calculate_mean_model_probability(y_hat_proba), 2
        )

        mcc = np.round(matthews_corrcoef(y, y_hat), 2)

        return precision, recall, f1, accuracy, mean_class_predicted_probability, mcc

    if model_type == "regressor":
        # regressor
        rmspe = root_mean_abs_percent_error(y, y_hat_proba)
        mape = mean_abs_percent_error(y, y_hat_proba)

        pep, plp = PEP_PLP(y, y_hat_proba)

        return rmspe, mape, pep, plp
