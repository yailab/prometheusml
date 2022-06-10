import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prometheus.algorithms.ensemble.gradient_boosting import (
    gradient_boosting_model,
    gradient_boosting_model_inference,
)
from prometheus.algorithms.ensemble.random_forest import RF_model, RF_model_inference
from prometheus.algorithms.linear_model.logistic_regression import (
    logistic_regression_model,
    logistic_regression_model_inference,
)
from prometheus.algorithms.svm.nu_svm import nu_svm_model, nu_svm_model_inference
from prometheus.algorithms.utils import create_input_output
from prometheus.data_processing.basic_data_checks import basic_data_processing
from prometheus.feature_extraction.general import data_feature_engineering
from prometheus.feature_ranking.feature_ranking_RF_CV import (
    feature_selection_with_RF_and_CV,
)
from prometheus.model_selection.model_cross_validation import (
    calculate_model_performance,
    cross_validate_the_models,
)
from prometheus.visualisation.inference import plot_prediction_classification
from prometheus.visualisation.validation import plot_prediction_classification_cv
from webapp import create_app
from webapp.models import User
from webapp.utils_task import (
    get_intermediate_path,
    get_uploaded_file_paths,
    read_uploaded_files,
)

# TODO: use type hints

# Background processes need some functionality existing in the app
# So we create a separate instance of the app to initialise the `Flask-SQLAlchemy` and get the configurations
# This is the only module that the RQ worker is going to import ??
app = create_app()
# because we are outside the initial app context we push the context
# (application object attributes, such as config) to the new app instance
# e.g. now `Flask-SQLAlchemy` can use the `current_app.config` to obtain their configuration.
app.app_context().push()


def data_process(user_id: int, project_id: int, asset: str, use_type: str) -> None:
    """
    Reads the uploaded data in the folder for the user and the project and does the
    automatic feature engineering.
    This function is running in a separate process controlled by RQ, not Flask.
    If an error occurs in the process controlled by the RQ, we log the error, along with the stack trace.
    :param user_id:
    :param project_id:
    :param asset:
    :param use_type: str (default: 'inference')
            Whether it is used in the training or inference phase.
    :return:
    """

    supported_asset = ["other"]
    if asset not in supported_asset:
        raise NotImplementedError

    # get the project from the db
    user = User.query.filter_by(id=user_id).first()
    project = user.projects.filter_by(id=project_id).first()

    # get the predictive target and features
    predictive_variable_name = project.get_tmp_train_pipeline().get("target_name")
    features = project.get_tmp_train_pipeline().get("user_selected_features")

    if user is None:
        raise ValueError("User not found in the db!")
    if project is None:
        raise ValueError("Project not found in the db!")

    try:
        # get the paths of the uploaded files
        uploaded_file_paths = get_uploaded_file_paths(user_id, project_id, use_type)
        intermediate_path = get_intermediate_path(user_id, project_id, use_type)

        # read the uploaded file(s)
        if use_type == "inference":
            use_cols = features
        else:
            use_cols = features + [predictive_variable_name]
        data_raw_list = read_uploaded_files(
            uploaded_file_paths,
            predictive_variable_name,
            use_type=use_type,
            use_cols=use_cols,
        )

        data_raw = data_raw_list[0]
        data_processed, _ = basic_data_processing(data_raw)  # run data processing

        # no tailored processing required so save dataset
        # -------------------------------------------------------------------- #
        # write the results to the filesystem
        processed_data_filename = os.path.join(
            intermediate_path, "processed_data.pickle"
        )
        with open(processed_data_filename, "wb") as handle:
            pickle.dump(data_processed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # -------------------------------------------------------------------- #
    except Exception:
        app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        raise


def feature_extract_manual(
    user_id: int,
    project_id: int,
    asset: str,
    use_type: str,
    features: list,
    transforms: list,
) -> None:
    """Task function to run the necessary feature transformations

    :param user_id:
    :param project_id:
    :param asset:
    :param use_type:
    :param features:
    :param transforms:
    :return:
    """
    supported_asset = ["other"]
    if asset not in supported_asset:
        raise ValueError(f"Asset {asset} not supported.")

    # get the project from the db
    user = User.query.filter_by(id=user_id).first()
    project = user.projects.filter_by(id=project_id).first()

    try:
        intermediate_path = get_intermediate_path(user_id, project_id, use_type)

        # Load the processed data dict from pickle
        processed_data_path = os.path.join(intermediate_path, "processed_data.pickle")
        # Load data (deserialize)
        with open(processed_data_path, "rb") as handle:
            data_unpickled = pickle.load(handle)

        # pass only the features after processing (some of them could be dropped)
        feature_set_after_process = set(data_unpickled.columns)
        feature_set_after_process.discard(
            "target"
        )  # remove the target from the set if it's present
        feature_input_set = set(features)

        if feature_set_after_process.issubset(
            feature_input_set
        ):  # validate that it's a subset of the user inputs
            feature_after_process = list(feature_set_after_process)
        else:
            raise ValueError(
                "The features after the data processing, are not a subset of the input features."
            )

        # get only the subset based on user selected features + target
        if use_type == "training":
            data_features = data_unpickled[feature_after_process + ["target"]]
        elif use_type == "inference":
            if "target" in data_unpickled.columns:
                data_features = data_unpickled[feature_after_process + ["target"]]
            else:
                data_features = data_unpickled[feature_after_process]
        else:
            raise ValueError(f"The model use type {use_type} not specified.")

        if use_type == "inference":
            # get the transformations from the training stage
            transforms = project.project_config.get("user_selected_transforms")

        # run the features' transformations
        data_features_extracted = data_feature_engineering(
            data_features, transforms, use_type=use_type
        )

        if use_type not in ["inference", "training"]:
            raise ValueError(f"The model use type {use_type} not specified.")

        # apply basic data processing after the feature extraction
        extracted_features_data_processed, _ = basic_data_processing(
            data_features_extracted
        )

        # write the extracted features after processing to the filesystem
        selected_extracted_features_filename = os.path.join(
            intermediate_path, "extracted_features_data_processed.csv"
        )
        extracted_features_data_processed.to_csv(selected_extracted_features_filename)

        # write the results to the filesystem
        # should we have both the processed and unprocessed data ??
        extracted_filename = os.path.join(
            intermediate_path, "extracted_features_data.csv"
        )
        data_features_extracted.to_csv(extracted_filename, index=False)

        # apply basic data processing after the feature extraction
        data_features_extracted, _ = basic_data_processing(data_features_extracted)
    except Exception:
        app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        raise


def feature_ranking_manual(user_id: int, project_id: int) -> tuple:
    try:
        intermediate_path = get_intermediate_path(user_id, project_id, "training")

        # Load the engineered features into a pandas df
        engineered_features_path = os.path.join(
            intermediate_path, "extracted_features_data.csv"
        )
        # TODO: note the index is assigned to False - change this if user passes index
        engineered_features_data = pd.read_csv(
            engineered_features_path, index_col=False
        )

        # no of features to display in the ranking plot
        no_of_display_features = 25

        # run feature ranking
        (
            f_importance_dataframe,
            figure_platform,
            figure_report,
        ) = feature_selection_with_RF_and_CV(
            engineered_features_data,
            display_features=no_of_display_features,
            alg_type="classifier",
        )

        # get the list of ranked features
        ranked_feature_list = f_importance_dataframe.feature_name.tolist()[
            :no_of_display_features
        ]

        # save the feature ranking plot for the platform
        # The media folder directory - for the platform display and reports
        media_folder = os.path.join(app.config["UPLOAD_FOLDER"], "media")
        Path(media_folder).mkdir(
            parents=True, exist_ok=True
        )  # create the directory if it doesn't exist

        filename_list = [str(user_id), str(project_id), "ranking", "platform.svg"]
        filename = "_".join(filename_list)
        export_path_fig = os.path.join(app.config["UPLOAD_FOLDER"], "media", filename)
        # save figure for the platform
        figure_platform.savefig(export_path_fig, transparent=True)

        # total number of features -- no cap for now
        no_features_max = engineered_features_data.shape[1]

        return no_features_max, ranked_feature_list

    except Exception:
        app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        raise


def model_training(
    user_id: int,
    project_id: int,
    algorithm: str,
    input_features: list,
    model_update: bool = False,
    **param_kwargs,
):
    """
    :param input_features:
    :param algorithm:
    :param user_id: int
        The id of the user
    :param project_id: int
        The id of the project
    :param model_update: bool
        Whether or not the function for first-time training (`True`) or model update (`False`)
    :return:
    """
    if model_update:
        pass
    else:
        try:
            intermediate_path = get_intermediate_path(user_id, project_id, "training")
            # Create the list of the uploaded file paths
            train_data_path = os.path.join(
                intermediate_path, "extracted_features_data_processed.csv"
            )
            # load the current (new) engineered data
            data = pd.read_csv(train_data_path, index_col=False)

            # overwrite the input dataset based on user selected top features
            target = "target"
            features_for_training = input_features + [target]
            data = data[features_for_training]

            # save dataset after user selects the top features
            data_extracted_features_top_path = os.path.join(
                intermediate_path, "data_extracted_features_top.csv"
            )
            data.to_csv(data_extracted_features_top_path)

        except Exception:
            app.logger.error("Unhandled exception", exc_info=sys.exc_info())
            raise

    # Split the dataset to features, regressor label, and classifier label
    X_train, y_train, group = create_input_output(data, output="target")

    # Train the regressor model and the classifier
    if algorithm == "rf":
        model, model_default = RF_model(
            X_train,
            y_train,
            cv_strategy="simple",
            scale_data=True,
            tuning_strategy="RGS",
            model_type="classifier",
            **param_kwargs,
        )
    elif algorithm == "nu_svc":
        model, model_default = nu_svm_model(
            X_train,
            y_train,
            scale_data=True,
            cv_strategy="simple",
            tuning_strategy="RGS",
            model_type="classifier",
            **param_kwargs,
        )
    elif algorithm == "logistic_regression":
        model, model_default = logistic_regression_model(
            X_train,
            y_train,
            scale_data=True,
            cv_strategy="simple",
            tuning_strategy="RGS",
            model_type="classifier",
            **param_kwargs,
        )
    elif algorithm == "gradient_boosting":
        model, model_default = gradient_boosting_model(
            X_train,
            y_train,
            scale_data=True,
            cv_strategy="simple",
            tuning_strategy="RGS",
            model_type="classifier",
            **param_kwargs,
        )
    else:
        raise NotImplementedError("Algorithm not supported.")

    return model, model_default


def model_validation(user_id: int, project_id: int) -> dict:
    """Function to produce the results of the model validation for visualisation

    :param user_id: int
        The id of the user
    :param project_id: int
        The id of the project
    :return:
    """
    # Get the user and the project
    user = User.query.filter_by(id=user_id).first()
    project = user.projects.filter_by(id=project_id).first()

    # Get the training data
    # -------------------------------------------------------------------- #
    try:
        intermediate_path = get_intermediate_path(user_id, project_id, "training")
        # Create the list of the uploaded file paths
        train_data_path = os.path.join(
            intermediate_path, "data_extracted_features_top.csv"
        )
        data = pd.read_csv(train_data_path, index_col=0)
    except OSError as e:
        app.logger.exception("Filesystem error: %s" % e)
        raise e

    # Load the models from the database
    pipe_tuned = pickle.loads(project.model_pipeline)
    pipe_default = pickle.loads(project.model_pipeline_default)

    # Split the dataset to features, regressor label, and classifier label
    X_train, y_train, group = create_input_output(data, model_type="classifier")

    # cross-validate both the tuned model and the default model
    pipe_list = [pipe_tuned, pipe_default]
    pipe_name_list = ["tuned", "default"]
    return_results = []
    for name, pipe in zip(pipe_name_list, pipe_list):
        # cross-validate model
        y_hat, y_hat_proba = cross_validate_the_models(
            data, pipe, model_type="classifier", decision_threshold=0.55
        )

        # ===================== Plots ======================== #
        # for the selected decision threshold(s)
        d_th = 0.55

        # get the classification performance metrics
        (
            precision,
            recall,
            f1,
            accuracy,
            mean_class_predicted_probability,
            matthews_corrcoef,
        ) = calculate_model_performance(
            y_train, y_hat_proba, model_type="classifier", decision_threshold=d_th
        )

        # create the validation plot(s)
        (
            fig_confusion_matrix,
            fig_precision_recall,
        ) = plot_prediction_classification_cv(
            y=y_train, y_hat_proba=y_hat_proba, decision_threshold=d_th
        )

        # Save the plots
        media_folder = os.path.join(app.config["UPLOAD_FOLDER"], "media")
        Path(media_folder).mkdir(
            parents=True, exist_ok=True
        )  # create the directory if it doesn't exist

        # save the confusion matrix plot for the platform
        filename_conf_list = [
            str(user_id),
            str(project_id),
            "validation",
            "conf-" + str(int(d_th * 100)),
            name,
            "platform.svg",
        ]
        filename_conf = "_".join(filename_conf_list)
        media_path_conf_fig = os.path.join(media_folder, filename_conf)
        fig_confusion_matrix.savefig(media_path_conf_fig, transparent=True)

        # save the precision-recall plot for the platform
        filename_prec_recall_list = [
            str(user_id),
            str(project_id),
            "validation",
            "recall-" + str(int(d_th * 100)),
            name,
            "platform.svg",
        ]
        filename_prec_recall = "_".join(filename_prec_recall_list)
        media_path_recall_fig = os.path.join(media_folder, filename_prec_recall)
        fig_precision_recall.savefig(media_path_recall_fig, transparent=True)

        # create the dictionary
        dct_performance = {
            "decision_threshold": np.round(d_th, 2),
            "metrics": {
                "precision": {"value": float(precision), "desc": "Precision"},
                "recall": {"value": float(recall), "desc": "Sensitivity"},
                "f1": {"value": float(f1), "desc": "F1 score"},
                "accuracy": {"value": float(accuracy), "desc": "Balanced accuracy"},
                "mean_class_predicted_prob": {
                    "value": float(mean_class_predicted_probability),
                    "desc": "Average class probability ",
                },
                "matthews_corrcoef": {
                    "value": float(matthews_corrcoef),
                    "desc": "Model quality (Phi coefficient) [-1, 1]",
                },
            },
        }

        # append the list of dicts
        return_results.append(dct_performance)

    # create a dict with the results for each model
    metrics_dct = {key: value for (key, value) in zip(pipe_name_list, return_results)}

    return metrics_dct


def model_inference(user_id: int, project_id: int) -> None:
    """Function that does the inference."""
    try:
        # Load the user and the project
        user = User.query.filter_by(id=user_id).first()
        project = user.projects.filter_by(id=project_id).first()

        # Load the model from the db, based on the user selection
        selected_model = project.project_config.get("user_selected_model")
        if selected_model == "tuned":
            model = pickle.loads(project.model_pipeline)  # unpickle the selected model
        elif selected_model == "default":
            model = pickle.loads(
                project.model_pipeline_default
            )  # unpickle the selected model
        else:
            raise ValueError(f"Model '{selected_model}' not supported.")

        intermediate_path = get_intermediate_path(user_id, project_id, "inference")

        # Load the engineered features into a pandas df
        engineered_features_path = os.path.join(
            intermediate_path, "extracted_features_data_processed.csv"
        )
        engineered_features_data = pd.read_csv(engineered_features_path, index_col=0)

        # Subset the data on the engineered features selected during the training phase
        selected_engineered_features_list = project.project_config.get(
            "user_selected_engineered_features"
        )
        engineered_features_data = engineered_features_data[
            selected_engineered_features_list
        ]

        # compute the X_test based on trained model features
        X_test = engineered_features_data

        decision_threshold = project.project_config.get("decision_threshold")
        # Predict
        if project.algorithm == "rf":
            results_dictionary = RF_model_inference(
                X_test,
                model=model,
                model_type="classifier",
                decision_threshold=decision_threshold,
            )
        elif project.algorithm == "nu_svc":
            results_dictionary = nu_svm_model_inference(
                X_test,
                model=model,
                model_type="classifier",
                decision_threshold=decision_threshold,
            )
        elif project.algorithm == "logistic_regression":
            results_dictionary = logistic_regression_model_inference(
                X_test,
                model=model,
                model_type="classifier",
                decision_threshold=decision_threshold,
            )
        elif project.algorithm == "gradient_boosting":
            results_dictionary = gradient_boosting_model_inference(
                X_test,
                model=model,
                model_type="classifier",
                decision_threshold=decision_threshold,
            )
        else:
            raise NotImplementedError("Algorithm not supported.")

        # compute the results figure for the platform
        # TODO pass the target name here
        target = project.project_config.get("target")
        figure_results = plot_prediction_classification(
            X_test, results_dictionary, target_name=target, report_type="platform"
        )

        media_folder = os.path.join(app.config["UPLOAD_FOLDER"], "media")
        Path(media_folder).mkdir(
            parents=True, exist_ok=True
        )  # create the directory if it doesn't exist

        export_folder = os.path.join(app.config["UPLOAD_FOLDER"], "exports")
        Path(export_folder).mkdir(
            parents=True, exist_ok=True
        )  # create the directory if it doesn't exist

        filename_list = [str(user_id), str(project_id), "inference", "platform.svg"]
        filename = "_".join(filename_list)
        media_path_fig = os.path.join(media_folder, filename)
        # save figure for the platform
        figure_results.savefig(media_path_fig, transparent=True)

        # concatenate the features df and the results
        results_df = pd.concat(
            [X_test.reset_index(), pd.DataFrame.from_dict(results_dictionary)], axis=1
        )

        # save the results' data to export
        filename_pred_list = [str(user_id), str(project_id), "prediction_results.csv"]
        filename_pred = "_".join(filename_pred_list)
        export_path_csv = os.path.join(export_folder, filename_pred)
        results_df.to_csv(export_path_csv, index=False)

    except Exception:
        app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        raise
