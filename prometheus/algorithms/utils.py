"""A collection of utilities functions for predictive algorithms"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


def create_input_output(data, output="target", model_type="classifier"):
    """Function that creates the input and label vectors for both the regressor and the classifier

    :param data: pandas dataframe
    :param model_type: choose if the algorithm is a classifier or regressor
    :param output: name of the label
    :return:
    """
    # Function that creates the input and output for the model

    # features to drop from X
    if "group" in data.columns:
        group = data.group
        features_to_drop_input = [
            output,
            "group",
        ]  # drop the label from the training dataset

    else:
        group = None
        features_to_drop_input = [output]  # drop the label from the training dataset

    if model_type == "classifier":
        X = data.drop(features_to_drop_input, axis=1, errors="ignore")
        # compute the label
        y = data[output]  # label
    elif model_type == "regressor":
        # seems to be the same as classifier - Do we need to have one for both classifier/regressor?
        X = data.drop(features_to_drop_input, axis=1, errors="ignore")
        # compute the label
        y = data[output]  # label
    else:
        raise ValueError("No model type has been selected!")

    return X, y, group


def gini_normalized(model, X, y_actual):
    """
    Classifier ONLY
    Function that calculates the normalise gini value for loss purposes during training

    Simple normalized Gini based on Scikit-Learn's roc_auc_score. For more on the
    implementation see here:
    https://www.kaggle.com/mathcass/tips-for-using-scikit-learn-for-evaluation

    :param model: the scikit model
    :param X_test: feature input matrix
    :param y_actual: true label
    :return:
    """
    # If the predictions y_pred are binary class probabilities

    y_pred = model.predict(X)

    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]

    def gini(a, p):
        return 2 * roc_auc_score(a, p) - 1

    return gini(y_actual, y_pred) / gini(y_actual, y_actual)


def compute_dataset_imbalance(X_train, y_train):
    """
    Function to calculate the model class_weight for imbalanced datasets using RF
    """

    weights = np.linspace(0.05, 0.95, 20)

    alg = RandomForestClassifier()

    gsc = GridSearchCV(
        estimator=alg,
        param_grid={"class_weight": [{0: x, 1: 1.0 - x} for x in weights]},
        scoring="f1",
        cv=3,
    )
    grid_result = gsc.fit(X_train, y_train)

    return grid_result.best_params_["class_weight"]


def compute_class(y_hat_proba, decision_threshold):
    """Compute hard-class based on the decion threshold
    Note: to be used for binary problems only
    """

    y_hat = []  # compute prediction
    for i in y_hat_proba.flatten()[1::2]:

        if i < decision_threshold:
            y_hat.append(0)
        else:
            y_hat.append(1)

    return np.array(y_hat)
