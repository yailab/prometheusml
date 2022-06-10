from random import seed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV

from prometheus.algorithms.ensemble.random_forest import RF_model
from prometheus.algorithms.utils import create_input_output, gini_normalized

seed(42)
np.random.seed(42)

"""
Pipeline Input:
1. Tune RF model on feature selection datasets
2. Use tuned model in RFE are refit on data
3. Unsupervised selection of important features via RFECV-RF
4. Create new training dataset (.transform) are prepare training dataset
5. Create new test set using the features determined by the pipeline

Pipeline Output:
- optimum number of features
- ranking of features it list format [rank, feature]
-  input training dataset as pandas with optimum no. of features
- input test dataset as pandas with optimum no. of features
- target variable training dataset
- target variable test dataset
"""

# name for predictive variable
# TODO: to be removed
predictive_variable_name = "target"


__all__ = ["feature_selection_RFE_CV"]


def _create_tune_model(X, y, **kwargs):

    model = RF_model(
        X, y, automatic=True, tuning_strategy="RGS", model_type="classifier", **kwargs
    )

    return model


def feature_selection_RFE_CV(data):
    no_of_features = 1  # the number of features to eliminate at every step

    # create input and output for the model
    X_train, y_train, group = create_input_output(
        data, model_type="classifier", output=predictive_variable_name
    )

    # Hyper-param tune Random Forest
    param_update = {"max_features": [no_of_features]}
    model = _create_tune_model(X_train, y_train, **param_update)

    # '''Recurrent Feature Elimination'''
    names = data.columns

    rfe = RFECV(
        estimator=model,
        min_features_to_select=no_of_features,
        cv=5,
        step=1,
        scoring=gini_normalized,
        verbose=2,
    )

    selector_RF = rfe.fit(X_train, y_train)

    ranking_features = sorted(
        zip(map(lambda x: round(x, 4), selector_RF.ranking_), names), reverse=False
    )
    optimal_no_feature = selector_RF.n_features_

    x = range(no_of_features, len(selector_RF.grid_scores_) + no_of_features)
    y = selector_RF.grid_scores_

    """feature selection resuts"""
    print("Feature rank: \n {}".format(ranking_features))
    # Plot number of features VS. cross-validation scores
    f = plt.figure(figsize=(7, 5))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(x, y, "o--", color="tab:orange")
    plt.plot(x[np.argmax(y)], np.max(y), "v", markersize=15, color="k")
    plt.xlabel("Selected no. of features", fontsize=15)
    plt.ylabel("Cross-validation score [Negative MSE]", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(False)
    plt.show()

    return (
        f,
        optimal_no_feature,
        ranking_features,
    )
