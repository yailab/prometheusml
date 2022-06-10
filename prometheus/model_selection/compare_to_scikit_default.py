"""Function to check model performance to scikit default"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from prometheus.metrics import root_mean_abs_percent_error


def compare_to_scikit_default(tuned_model, default_model, X, y, model_type):
    """ """

    # trained model - make pipeline with best estimator
    new_pipe_list = tuned_model.steps[:-1]
    new_pipe_list.extend(
        [("algorithm", tuned_model.named_steps.get("search_strategy").best_estimator_)]
    )
    p = Pipeline(new_pipe_list)
    model_from_tuning = p.fit(X, y)

    # default model
    model_from_default = default_model.fit(X, y)

    # predict with the model
    y_hat_tuned = model_from_tuning.predict(X)
    y_hat_scikit_default = model_from_default.predict(X)

    # specify the accuracy metric based on model_type
    if model_type == "regressor":
        accuracy_metric = root_mean_abs_percent_error

    elif model_type == "classifier":
        accuracy_metric = f1_score

    else:
        raise ValueError(
            "Model comparison to scikit-default failed - missing model_type"
        )

    acc_tuned = np.round(accuracy_metric(y_hat_tuned, y_hat_tuned) * 100, 2)
    acc_scikit_default = np.round(
        accuracy_metric(y_hat_scikit_default, y_hat_tuned) * 100, 2
    )

    if acc_tuned <= acc_scikit_default:
        # model_to_return = model_from_scikit
        print("Model with default parameters has higher accuracy!")
    elif acc_tuned > acc_scikit_default:
        # model_to_return = model_from_tuning
        print("Model with tuned parameters has higher accuracy!")
    else:
        raise ValueError("")

    return model_from_tuning, model_from_default
