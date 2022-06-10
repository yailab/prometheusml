from __future__ import annotations

import os
import pickle
import shutil
import sys
import uuid
from functools import partial

import numpy as np
import pandas as pd
import redis
import yaml
from flask import (
    abort,
    current_app,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask.typing import ResponseReturnValue
from flask_login import current_user, login_required
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from sklearn.pipeline import Pipeline
from werkzeug.utils import secure_filename

from webapp import db
from webapp.main import bp
from webapp.utils import is_numeric_data
from webapp.utils_task import get_uploaded_file_paths

# ==================================================================== #
#                        Views templates                               #
# ==================================================================== #


# main page
@bp.route("/index")
@login_required
def index() -> ResponseReturnValue:
    return render_template("index.html", title="Home")


# user profile page
@bp.route("/user")
@login_required
def user() -> ResponseReturnValue:
    """
    Get the current user main page.
    """
    projects = current_user.projects.all()

    return render_template("user.html", projects=projects)


# projects list page
@bp.route("/projects")
@login_required
def projects() -> ResponseReturnValue:
    """
    Get the projects of the user.
    """
    projects = current_user.projects.all()

    return render_template("projects_home.html", projects=projects)


# login page
@bp.route("/")
def starting_page() -> ResponseReturnValue:
    return redirect(url_for("auth.login"))


# Training views
@bp.route("/projects/<int:project_id>/input/update")
@login_required
def project_input_update(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    return render_template("training/input.html", project=project)


@bp.route("/projects/<int:project_id>/transform/update")
@login_required
def data_transform_update(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    return render_template(
        "training/transform_data.html", user=current_user, project=project
    )


@bp.route("/projects/<int:project_id>/algorithm/update")
@login_required
def algorithm_update(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    return render_template(
        "training/ai_algorithm.html", user=current_user, project=project
    )


@bp.route("/projects/<int:project_id>/validate/update")
@login_required
def model_validate_update(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    return render_template(
        "training/validation.html", user=current_user, project=project
    )


@bp.route("/projects/<int:project_id>/model/update")
@login_required
def model_update(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    return render_template(
        "training/model_update.html", user=current_user, project=project
    )


# inference views
@bp.route("/projects/<int:project_id>/input")
@login_required
def project_input(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    return render_template("inference/input.html", project=project)


@bp.route("/projects/<int:project_id>/predict")
@login_required
def data_predict(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    return render_template("inference/predict.html", user=current_user, project=project)


# ==================================================================== #
#                        Functional routes                             #
# ==================================================================== #
@bp.post("/check_uploads/<int:project_id>")
@login_required
def check_uploads_inference(project_id: int) -> ResponseReturnValue:
    # Get the existing features list
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    current_feature_list = project.get_tmp_train_pipeline().get(
        "user_selected_features"
    )

    uploaded_file_paths = get_uploaded_file_paths(
        user_id=current_user.id, project_id=project_id, use_type="inference"
    )

    # read the uploaded data
    uploaded_data = pd.read_csv(uploaded_file_paths[0], nrows=2)

    print(
        f"uploaded dataset features: \n {uploaded_data.columns.tolist()}",
        file=sys.stderr,
    )

    print(f"current features: \n {current_feature_list}", file=sys.stderr)

    # if current features not a subset
    if not set(current_feature_list).issubset(uploaded_data.columns.tolist()):
        # Get the missing features -- difference of the sets
        current_feature_set = set(current_feature_list)
        uploaded_feature_set = set(uploaded_data.columns.tolist())
        missing_feature_set = current_feature_set - uploaded_feature_set
        rtn_dict = {
            "errorMessage": f"The uploaded dataset does not have all available features. "
            f"The following are missing: {missing_feature_set}"
        }
        return jsonify(rtn_dict), 501
    else:
        return make_response(("Uploading data for inference successful", 200))


@bp.post("/get_app_areas")
@login_required
def get_app_areas() -> ResponseReturnValue:
    # read the yaml config file
    root_path_app = os.path.dirname(current_app.instance_path)
    app_area_config_path = os.path.join(
        root_path_app, "configs/config_supported_app_areas.yml"
    )
    with open(app_area_config_path, "r") as stream:
        try:
            app_area_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            current_app.logger.exception("YAML error: %s" % exc)

    # Parse the yaml file
    app_areas_dict = app_area_config["app_areas"]

    labels_list = [values["label"] for keys, values in app_areas_dict.items()]
    id_list = [values["id"] for keys, values in app_areas_dict.items()]

    return jsonify({"id_list": id_list, "labels_list": labels_list})


@bp.post("/upload_input_data/<int:project_id>/<string:dataset_type>")
@login_required
def upload_input(project_id: int, dataset_type: str) -> ResponseReturnValue:
    """Get the uploaded file when using chunked files

    It uses uuid to tell apart the different files.

    Arg
    :param project_id:
        The id of the current project
    :param dataset_type:
        The type of dataset. It's either `training` or `inference`
    """
    uploaded_file = request.files.get("file")
    # Sanitise the filename provided by the client
    filename = secure_filename(uploaded_file.filename)

    # if they have uploaded something
    if filename != "":
        # split the filename to the name and the extension
        file_ext = os.path.splitext(filename)[1]
        # Check that the uploaded file is one of the allowed file extensions
        if file_ext not in current_app.config["ALLOWED_EXTENSIONS"]:
            # or file_ext != validate_csv(uploaded_file.stream):
            return "Invalid file type", 400

        # -------------------------------------------------------------------- #
        # create the intermediate path
        # if the file is NOT chunked
        root_path_app = os.path.dirname(current_app.instance_path)
        if "dzchunkindex" not in request.form.keys():
            intermediate_path = os.path.join(
                root_path_app,
                current_app.config["UPLOAD_FOLDER"],
                str(dataset_type),
                str(current_user.get_id()),
                str(project_id),
                str(uuid.uuid4()),
            )
        else:
            intermediate_path = os.path.join(
                root_path_app,
                current_app.config["UPLOAD_FOLDER"],
                str(dataset_type),
                str(current_user.get_id()),
                str(project_id),
                str(request.form.get("dzuuid")),
            )
        # Create the file path if it doesn't exist
        if not os.path.exists(intermediate_path):
            os.makedirs(intermediate_path)

        # Create the new name based on the user_id, project_id, and
        custom_filename = (
            "uploaded_dataset_"
            + str(current_user.get_id())
            + "_"
            + str(project_id)
            + ".csv"
        )

        # The uploaded file path
        upload_file_path = os.path.join(intermediate_path, custom_filename)
        # -------------------------------------------------------------------- #
        # if the file is NOT chunked
        if "dzchunkindex" not in request.form.keys():
            # save the file to the directory
            try:
                uploaded_file.save(upload_file_path)
            except OSError:
                # Log the error
                current_app.logger.exception("Could not write to file")
                return make_response(
                    ("Not sure why, but we couldn't write the file to disk", 500)
                )

            return make_response(("File(s) upload successful", 200))

        else:
            # If the file is chuncked
            # Get the current chunk from dropzone
            current_chunk = int(request.form["dzchunkindex"])

            # Handle the chunked file
            try:
                # append to the file, ans write as bytes --> 'ab' mode
                with open(upload_file_path, "ab") as f:
                    # use the offset of dropzone to write the new chunk after the already written ones
                    f.seek(int(request.form["dzchunkbyteoffset"]))
                    f.write(uploaded_file.stream.read())
            except OSError:
                # Log the error
                current_app.logger.exception("Could not write to file")
                return make_response(
                    ("Not sure why, but we couldn't write the file to disk", 500)
                )

            # Check that the whole file has been uploaded
            total_chunks = int(request.form["dztotalchunkcount"])
            # check that it was the last chunk
            if current_chunk + 1 == total_chunks:
                # check that the uploaded has the same size as from the POST request
                if os.path.getsize(upload_file_path) != int(
                    request.form["dztotalfilesize"]
                ):
                    # log an error if there is a size mismatch
                    current_app.logger.error(
                        f"File {filename} was completed, "
                        f"but has a size mismatch."
                        f"Was {os.path.getsize(upload_file_path)} but we"
                        f" expected {request.form['dztotalfilesize']} "
                    )

                    return make_response(("Size mismatch", 500))
                else:
                    current_app.logger.info(
                        f"File {filename} has been uploaded successfully"
                    )
            else:
                current_app.logger.debug(
                    f"Chunk {current_chunk + 1} of {total_chunks} "
                    f"for file {filename} complete"
                )

            return make_response(("Chunk upload successful", 200))


@bp.route(
    "/remove_input_data/<int:project_id>/<string:dataset_type>", methods=["GET", "POST"]
)
@login_required
def remove_upload_data(project_id: int, dataset_type: str) -> ResponseReturnValue:
    """Delete the uploaded data in the filesystem.

    :param project_id: str
        The id of the current project
    :param dataset_type: str
        The type of dataset. It's either `training` or `inference`
    :return:
    """

    # create the intermediate path
    root_path_app = os.path.dirname(current_app.instance_path)
    intermediate_path = os.path.join(
        root_path_app,
        current_app.config["UPLOAD_FOLDER"],
        str(dataset_type),
        str(current_user.get_id()),
        str(project_id),
    )
    if os.path.exists(intermediate_path):
        # clear the intermediate directory
        for filename in os.listdir(intermediate_path):
            file_path = os.path.join(intermediate_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except OSError as e:
                current_app.logger.exception(
                    "Failed to delete %s. Reason: %s" % (file_path, e)
                )
                return make_response(("Problem with uploaded file(s) deletion", 500))
    else:
        make_response(("No files to delete. Deletion skipped.", 200))

    return make_response(("File(s) deletion successful", 200))


# function to validate uploaded files
@bp.post("/validate_input/<int:project_id>")
@login_required
def validate_input(project_id: int) -> ResponseReturnValue:

    # get the paths of the uploaded files (only for training pipeline)
    uploaded_file_paths = get_uploaded_file_paths(
        current_user.id, project_id, "training"
    )

    print(f"The paths are: {uploaded_file_paths}", file=sys.stderr)

    # read the minimum acceptable number of rows [a single uploaded file]
    min_lines = 35
    df = pd.read_csv(uploaded_file_paths[0], nrows=min_lines)

    if df.shape[0] < min_lines:
        rtn_dict = {
            "message": f"The uploaded data samples are not enough. "
            f"Uploaded {df.shape[0]}, and need to have at least {min_lines} samples.",
            "status": "failed",
        }
        return jsonify(rtn_dict)
    else:
        return jsonify({"status": "success"})


@bp.post("/stop_task")
@login_required
def stop_job() -> ResponseReturnValue:
    """Route to stop a running task
    Returns:
        Message whether the task is stopped.
    """
    # Get the latest task id from the session -- the session is client-specific
    job_id = session.get("job_id")

    # Fetch the job with the specific id
    try:
        job = current_app.task_queue.fetch_job(job_id)

        if job:
            # get the dependent rq jobs and the current job
            jobs = [job] + job.fetch_dependencies()
            print(f"The current jobs to be stopped are: \n {jobs}", file=sys.stderr)

            # cancel all of them
            for job in jobs:
                job_status = job.get_status()  # get the job current status
                if job_status == "queued":
                    job.delete()  # remove it from the queue and free memory
                elif job_status == "started":
                    try:
                        send_stop_job_command(current_app.redis, job.get_id())
                    except InvalidJobOperation:
                        current_app.logger.error(
                            "Unhandled exception", exc_info=sys.exc_info()
                        )
                        return jsonify(
                            {
                                "errorMessage": "Couldn't stop the task. Contact yaiLab team."
                            }
                        )

            return jsonify({"message": "You successfully stopped Prometheus..."})
        else:
            return jsonify({"message": "No jobs found to stop..."})
    except (redis.exceptions.DataError, NoSuchJobError) as e:
        current_app.logger.exception("RQ error: %s" % e)
        return jsonify({"message": "No jobs found to stop..."})


@bp.route("/tasks/<string:task_id>/status", methods=["GET"])
@login_required
def get_status(task_id: str) -> ResponseReturnValue:
    # Fetch the job with the specific id
    task = current_app.task_queue.fetch_job(task_id)

    if task:
        response_object = {
            "status": "success",
            "data": {"task_id": task.get_id(), "task_status": task.get_status()},
        }
    else:
        response_object = {"status": "unknown"}

    return jsonify(response_object)


@bp.route("/tasks/<string:task_id>/result", methods=["GET"])
@login_required
def get_result(task_id: str) -> ResponseReturnValue:
    # Fetch the job with the specific id
    task = current_app.task_queue.fetch_job(task_id)

    if task:
        response_object = {
            "status": "success",
            "data": {"task_id": task.get_id(), "task_result": task.result},
        }
    else:
        response_object = {"status": "failed"}

    return jsonify(response_object)


@bp.get(
    "/get_img_src/<int:project_id>/<string:pipe_step>/<string:fig_type>/<string:model_type>"
)
@login_required
def get_img_src(
    project_id: int,
    pipe_step: str,
    fig_type: str | None = None,
    model_type: str | None = None,
) -> ResponseReturnValue:
    """Get the XML text of an SVG image"""
    if pipe_step not in ["ranking", "inference", "validation"]:
        raise ValueError(f"No svg image created for ML step {pipe_step}")

    # get the project from db
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # svg image path
    filename_list = [
        str(current_user.id),
        str(project.id),
        pipe_step,
        fig_type,
        model_type,
        "platform.svg",
    ]
    # remove None(s) from the list
    while "None" in filename_list:
        filename_list.remove("None")
    # create the proper filename
    filename = "_".join(filename_list)
    img_path = os.path.join(current_app.config["UPLOAD_FOLDER"], "media", filename)

    # read the svg image
    with open(img_path) as f:
        svg_img = f.read()

    rtn_dict = {"img_svg": svg_img}
    return jsonify(rtn_dict)


@bp.route("/export/<int:project_id>/<string:name>", methods=["GET", "POST"])
@login_required
def download_file(project_id: int, name: str):
    # get the project from db
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    export_folder = os.path.join(
        os.path.dirname(current_app.instance_path),
        current_app.config["UPLOAD_FOLDER"],
        "exports",
    )
    filename_list = [str(current_user.id), str(project.id), name + ".csv"]
    filename = "_".join(filename_list)
    file_path = os.path.join(export_folder, filename)
    download_filename = project.project_name + "-" + name + ".csv"

    try:
        return send_file(file_path, as_attachment=True, download_name=download_filename)
    except FileNotFoundError:
        abort(404, description="Resource not found")


# ================================================================================== #
#                                 TRAINING FUNCTIONALITY                             #
# ================================================================================== #
@bp.post("/get_uploaded_features/<int:project_id>")
@login_required
def get_features_input(project_id: int) -> ResponseReturnValue:
    """Get the features of the data uploaded by the user."""

    uploaded_file_paths = get_uploaded_file_paths(
        current_user.id, project_id, use_type="training"
    )

    read_csv_top_func = partial(pd.read_csv, nrows=100)  # get the list of pandas dfs
    read_csv_top_func.__doc__ = "Read only the 100 first rows for fast results"
    data_input_list_all = list(
        map(read_csv_top_func, uploaded_file_paths)
    )  # get list of input dataframes
    # keep only the numerical features
    data_numeric_list = [
        df.loc[:, df.apply(is_numeric_data)] for df in data_input_list_all
    ]
    # drop the columns that have only NaNs (for the first 100 rows)
    data_input_list = [df.dropna(axis=1, how="all") for df in data_numeric_list]
    # check that all the uploaded data have the same columns
    if all([data_input_list[0].shape[1] == df.shape[1] for df in data_input_list]):
        if all(
            [
                len(data_input_list[0].columns.intersection(df.columns))
                == data_input_list[0].shape[1]
                for df in data_input_list
            ]
        ):
            if all(
                [
                    all(data_input_list[0].columns == df.columns)
                    for df in data_input_list
                ]
            ):
                feature_col_list = data_input_list[0].columns.tolist()
                return jsonify({"feature_col_list": feature_col_list})
            else:
                return make_response(
                    (
                        "The uploaded datasets do not have the same sequence of features!",
                        403,
                    )
                )
        else:
            return make_response(
                ("The uploaded datasets don't have the same columns!", 403)
            )
    else:
        return make_response(
            ("The uploaded datasets don't have the same number of columns!", 403)
        )


@bp.post("/set_target/<int:project_id>")
@login_required
def set_target(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the target selected by the user
    target_name = request.get_json()["target_name"]

    # return an error when the user hasn't selected a target
    if not target_name:
        return make_response(
            ("The output variable wasn't registered, please try again!", 403)
        )

    # update the db (temporary column)
    project.add_update_tmp_train({"target_name": target_name})

    return jsonify({"status": f"Successfully got the target: {target_name}"})


@bp.post("/set_user_selected_features/<int:project_id>")
@login_required
def set_user_selected_features(project_id: int) -> ResponseReturnValue:

    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the list of features labels
    feature_label_json = request.get_json()
    feature_label_list = feature_label_json["user_selected_features"]

    # TODO: do some checks about the validity of the data received

    # update the db
    project.add_update_tmp_train({"user_selected_features": feature_label_list})

    return jsonify({"status": "Successfully got the user-selected features"})


@bp.post("/get_data_profile/<int:project_id>")
@login_required
def get_data_profile(project_id: int) -> ResponseReturnValue:
    # get the algorithm selected by the user
    selected_features = request.get_json()["user_selected_features"]

    print(
        f"The selected features for the report are: {selected_features}",
        file=sys.stderr,
    )

    # return an error when the user hasn't selected any features
    if not selected_features:
        return make_response(("There was no feature selected for data profiling!", 403))

    try:
        # run a background task for the data profiling
        task, _ = current_user.launch_task(
            name="data_profile",
            description="profiling of uploaded data",
            user_id=current_user.id,
            project_id=project_id,
            app_area="generic",
            selected_features=selected_features,
        )
        db.session.commit()  # commit the task to db
        session["job_id"] = task.id  # add job id to session
    except Exception:
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

    response_object = {"status": "success", "data": {"task_id": task.id}}
    return jsonify(response_object), 202


@bp.post("/get_transforms/<int:project_id>")
@login_required
def get_transform_list(project_id: int) -> ResponseReturnValue:
    # Get project
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # read the yaml config file
    root_path_app = os.path.dirname(current_app.instance_path)
    feature_eng_config_path = os.path.join(
        root_path_app, "configs/config_feature_engineering.yml"
    )

    with open(feature_eng_config_path, "r") as stream:
        try:
            feature_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            current_app.logger.exception("YAML error: %s" % exc)

    # Query the necessary parameters
    feature_app = feature_config["app_areas"].get(project.app_area)
    feature_asset = feature_app["assets"].get(project.asset)

    # return signals only if signal type (rolling window)
    if project.app_area not in ["continuous_prediction", "class_prediction"]:
        signals = [values for keys, values in feature_asset["signals"].items()]
    else:
        signals = []

    # get the transforms
    transforms = [values for keys, values in feature_asset["transforms"].items()]

    return jsonify({"signal_list": signals, "transform_list": transforms})


@bp.post("/run_transforms/<int:project_id>")
@login_required
def run_manual_transform(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the request json
    selected_transform_json = request.get_json()

    # get the list of the user selected transformations
    selected_signal_list = selected_transform_json["selected_signals"]
    selected_transform_list = selected_transform_json["selected_transforms"]

    # write them to the db
    project.add_update_tmp_train({"user_selected_signals": selected_signal_list})
    project.add_update_tmp_train({"user_selected_transforms": selected_transform_list})

    # if the list is non-empty
    if selected_transform_list:
        if project.asset == "battery":
            return jsonify(
                {
                    "info": f"Manual feature extraction not yet supported for {project.asset} asset...",
                    "progress": 0,
                }
            )
        elif project.asset in ["pump", "power_electronics", "other"]:
            # TODO: to be used in the progress bar
            try:
                task_process, job_process = current_user.launch_task(
                    name="data_process",
                    description="process-data",
                    user_id=current_user.id,
                    project_id=project_id,
                    app_area=project.app_area,
                    asset=project.asset,
                    use_type="training",
                )
                db.session.commit()  # commit the task to db
            except Exception:
                current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

            try:
                task, _ = current_user.launch_task(
                    name="feature_extract_manual",
                    description="manual feature extraction",
                    user_id=current_user.id,
                    project_id=project_id,
                    app_area=project.app_area,
                    asset=project.asset,
                    use_type="training",
                    features=project.get_tmp_train_pipeline().get(
                        "user_selected_features"
                    ),
                    transforms=project.get_tmp_train_pipeline().get(
                        "user_selected_transforms"
                    ),
                    job_timeout="1h",
                    depends_on=job_process,
                )
                db.session.commit()  # commit the task to db
                # add job id to session for stopping functionality
                session["job_id"] = task.id
            except Exception:
                current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        else:
            raise ValueError(f"The asset type {project.asset} is not supported yet!")

        response_object = {
            "status": "success",
            "data": {"task_id": task.id},
            "info": "Sit back and relax, Prometheus is extracting your features...",
            "pipeline": project.get_tmp_train_pipeline(),
        }
        return jsonify(response_object), 202

    else:
        # no feature engineering
        jsonify(
            {"info": "You have selected no features to engineer", "progress": 100}
        ), 200


@bp.post("/skip_feature_extraction/<int:project_id>/<string:use_type>")
@login_required
def skip_feature_extraction(project_id: int, use_type: str) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the request json
    selected_transform_json = request.get_json()

    # get the list of the user selected transformations
    selected_signal_list = selected_transform_json["selected_signals"]
    selected_transform_list = selected_transform_json["selected_transforms"]

    # write them to the db
    project.add_update_tmp_train({"user_selected_signals": selected_signal_list})
    project.add_update_tmp_train({"user_selected_transforms": selected_transform_list})

    try:
        task_process, job_process = current_user.launch_task(
            name="data_process",
            description="process the data",
            user_id=current_user.id,
            project_id=project_id,
            app_area=project.app_area,
            asset=project.asset,
            use_type=use_type,
        )
        db.session.commit()  # commit the task to db
    except Exception:
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

    try:
        task, _ = current_user.launch_task(
            name="feature_extract_manual",
            description="skip feature extraction",
            user_id=current_user.id,
            project_id=project_id,
            app_area=project.app_area,
            asset=project.asset,
            use_type="training",
            features=project.get_tmp_train_pipeline().get("user_selected_features"),
            transforms=project.get_tmp_train_pipeline().get("user_selected_transforms"),
            job_timeout="1h",
            depends_on=job_process,
        )
        db.session.commit()  # commit the task to db
        session["job_id"] = task.id  # add job id to session
    except Exception:
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

    response_object = {"status": "success", "data": {"task_id": task.id}}
    return jsonify(response_object), 202


@bp.post("/rank_features/<int:project_id>")
@login_required
def feature_ranking(project_id: int) -> ResponseReturnValue:

    project = current_user.projects.filter_by(id=project_id).first_or_404()

    if project.asset == "battery":
        return jsonify(
            {
                "info": f"Manual feature extraction not yet supported for {project.asset} asset...",
                "progress": 0,
            }
        )
    elif project.asset in ["pump", "power_electronics", "other"]:
        task, _ = current_user.launch_task(
            name="feature_ranking_manual",
            description="feature-ranking-training-purposes",
            user_id=current_user.id,
            project_id=project_id,
            app_area=project.app_area,
            job_timeout=3000,
        )
        db.session.commit()  # commit the task to db
        session["job_id"] = task.id  # add job id to session

        response_object = {
            "status": "success",
            "data": {"task_id": task.id},
            "info": "Sit back and relax, Prometheus is ranking your features...",
            "pipeline": project.get_tmp_train_pipeline(),
        }
        return jsonify(response_object), 202

    else:
        raise NotImplementedError(
            f"{project.asset} not yet supported for feature ranking"
        )


@bp.post("/set_ranked_features/<int:project_id>")
@login_required
def set_ranked_features(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    # get the list of user selected features after ranking
    request_json = request.get_json()
    ranked_features = request_json["ranked_feature_list"]

    print(f"The selected feature list is: \n{ranked_features}", file=sys.stderr)

    # write the selected features after ranking to the db
    project.add_update_tmp_train({"ranked_features": ranked_features})

    return jsonify({"message": f"Successfully got top features: {ranked_features}"})


@bp.post("/send_top_features/<int:project_id>")
@login_required
def set_top_features(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    # get the list of user selected features after ranking
    request_json = request.get_json()
    no_top_features = request_json["no_top_features_user_selected"]

    # get the ranked feature list
    ranked_feature_list = project.get_tmp_train_pipeline().get("ranked_features")
    ranked_feature_list_user_input = ranked_feature_list[:no_top_features]

    print(
        f"The selected feature list is: \n{ranked_feature_list_user_input}",
        file=sys.stderr,
    )

    # write the selected features after ranking to the db
    project.add_update_tmp_train(
        {"user_selected_features_after_ranking": ranked_feature_list_user_input}
    )

    return jsonify(
        {"status": f"Successfully got top features: {ranked_feature_list_user_input}"}
    )


@bp.post("/get_algorithms/<int:project_id>")
@login_required
def get_algo_list(project_id: int) -> ResponseReturnValue:
    # Get project
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # read the yaml config file
    root_path_app = os.path.dirname(current_app.instance_path)
    algo_config_path = os.path.join(root_path_app, "configs/config_algos_sklearn.yml")
    with open(algo_config_path, "r") as stream:
        try:
            algo_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            current_app.logger.exception("YAML error: %s" % exc)

    # Query the necessary parameters
    algo_app = algo_config["app_areas"].get(project.app_area)
    algo_asset = algo_app["assets"].get(project.asset)
    algo_dict = algo_asset["algorithms"]

    algo_label_list = [values["label"] for keys, values in algo_dict.items()]
    algo_id_list = [values["id"] for keys, values in algo_dict.items()]
    algo_desc_list = [values["description"] for keys, values in algo_dict.items()]
    algo_href_list = [values["href"] for keys, values in algo_dict.items()]

    rtn_dict = {
        "id_list": algo_id_list,
        "labels_list": algo_label_list,
        "desc_list": algo_desc_list,
        "href_list": algo_href_list,
    }

    return jsonify(rtn_dict)


@bp.post("/get_search_strategies/<int:project_id>")
@login_required
def get_hyperparam_search_list(project_id: int) -> ResponseReturnValue:
    # Get project
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # read the yaml config file
    root_path_app = os.path.dirname(current_app.instance_path)
    algo_config_path = os.path.join(
        root_path_app, "configs/config_hyperparam_search.yml"
    )
    with open(algo_config_path, "r") as stream:
        try:
            algo_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            current_app.logger.exception("YAML error: %s" % exc)

    # Query the necessary parameters
    algo_app = algo_config["app_areas"].get(project.app_area)
    algo_asset = algo_app["assets"].get(project.asset)
    algo_dict = algo_asset["search_strategies"]

    algo_label_list = [values["label"] for keys, values in algo_dict.items()]
    algo_id_list = [values["id"] for keys, values in algo_dict.items()]

    rtn_dict = {"id_list": algo_id_list, "labels_list": algo_label_list}

    return jsonify(rtn_dict)


@bp.post("/get_algo_params/<int:project_id>")
@login_required
def get_algo_params_list(project_id: int) -> ResponseReturnValue:
    # Get project
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # read the yaml config file
    root_path_app = os.path.dirname(current_app.instance_path)
    algo_config_path = os.path.join(root_path_app, "configs/config_algos_sklearn.yml")
    with open(algo_config_path, "r") as stream:
        try:
            algo_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            current_app.logger.exception("YAML error: %s" % exc)

    # Query the necessary parameters
    selected_algorithm = project.get_tmp_train_pipeline().get("user_selected_algorithm")
    algo_app = algo_config["app_areas"].get(project.app_area)
    algo_asset = algo_app["assets"].get(project.asset)
    algo_type = algo_asset["algorithms"].get(selected_algorithm)
    algo_params_dict = algo_type["parameters"]

    labels_params = [values["label"] for keys, values in algo_params_dict.items()]
    info_params = [values["info"] for keys, values in algo_params_dict.items()]
    min_params = [values["min"] for keys, values in algo_params_dict.items()]
    max_params = [values["max"] for keys, values in algo_params_dict.items()]
    from_params = [values["from"] for keys, values in algo_params_dict.items()]
    to_params = [values["to"] for keys, values in algo_params_dict.items()]
    step_params = [values["step"] for keys, values in algo_params_dict.items()]
    scale_params = [values["scale"] for keys, values in algo_params_dict.items()]
    keys_params = list(algo_params_dict.keys())

    rtn_dict = {
        "labels_list": labels_params,
        "min_list": min_params,
        "max_list": max_params,
        "step_list": step_params,
        "id_list": keys_params,
        "from_list": from_params,
        "to_list": to_params,
        "scale_list": scale_params,
        "info_list": info_params,
    }

    return jsonify(rtn_dict)


@bp.post("/get_hyper_params/<int:project_id>")
@login_required
def get_hyper_params_list(project_id: int) -> ResponseReturnValue:
    # Get project
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # read the yaml config file
    root_path_app = os.path.dirname(current_app.instance_path)
    algo_config_path = os.path.join(
        root_path_app, "configs/config_hyperparam_search.yml"
    )
    with open(algo_config_path, "r") as stream:
        try:
            algo_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            current_app.logger.exception("YAML error: %s" % exc)

    # Query the necessary parameters
    selected_hyper = project.get_tmp_train_pipeline().get(
        "user_selected_search_strategy"
    )
    algo_app = algo_config["app_areas"].get(project.app_area)
    algo_asset = algo_app["assets"].get(project.asset)
    algo_type = algo_asset["search_strategies"].get(selected_hyper)
    algo_params_dict = algo_type["parameters"]

    labels_params = [values["label"] for keys, values in algo_params_dict.items()]
    min_params = [values["min"] for keys, values in algo_params_dict.items()]
    max_params = [values["max"] for keys, values in algo_params_dict.items()]
    keys_params = list(algo_params_dict.keys())
    value_params = [values["value"] for keys, values in algo_params_dict.items()]

    rtn_dict = {
        "labels_list": labels_params,
        "id_list": keys_params,
        "min_list": min_params,
        "max_list": max_params,
        "value_list": value_params,
    }

    return jsonify(rtn_dict)


@bp.route("/<int:project_id>/train", methods=["POST"])
@login_required
def train_model(project_id: int) -> ResponseReturnValue:
    """Run the background task of training the models

    :param project_id: str
        The id of the current project
    """
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the names of the selected features after ranking
    top_selected_features = project.get_tmp_train_pipeline().get(
        "user_selected_features_after_ranking"
    )

    # get the params values from the user
    request_json = request.get_json()

    # When a request has empty data then use the automatic (do not update params)
    if not request_json:
        algo_params_dict = {}
        hyper_param_dict = {}
    else:
        # Get the params list from the user and construct the dict
        algo_params_list = request_json["params_names_values"]
        hyper_param_list = request_json["hyper_params_names_values"]

        hyper_param_dict = {}
        for d in hyper_param_list:
            hyper_param_dict[d["param_name"]] = d["param_value"]

        algo_params_dict = {}
        for d in algo_params_list:
            # decision tree based parameters
            if d["param_name"] == "n_estimators":
                stride = 10
            elif d["param_name"] == "max_features":
                d["param_min"] = min(d["param_min"], len(top_selected_features))
                d["param_max"] = min(d["param_max"], len(top_selected_features))
                stride = 1

            # nu-svm parameters
            elif d["param_name"] == "nu":
                stride = 0.1
            elif d["param_name"] == "C":
                stride = 0.2

            else:
                stride = 1

            if d["param_min"] == d["param_max"]:
                algo_params_dict[d["param_name"]] = np.array([d["param_min"]])
            elif d["param_name"] in [
                "alpha_1",
                "alpha_2",
                "lambda_1",
                "lambda_1",
            ]:  # special case for Bayesian Ridge
                algo_params_dict[d["param_name"]] = (
                    10
                    ** np.arange(
                        1 + abs(np.log10(d["param_max"]) - np.log10(d["param_min"]))
                    )
                    * d["param_min"]
                )
            else:
                algo_params_dict[d["param_name"]] = np.arange(
                    d["param_min"], d["param_max"], stride
                )

    # have them in a nested dict
    params_dict = {"algo_params": algo_params_dict, "hyper_params": hyper_param_dict}

    try:
        task, _ = current_user.launch_task(
            name="model_training",
            description="training of model(s)",
            user_id=current_user.id,
            project_id=project_id,
            app_area=project.app_area,
            algorithm=project.get_tmp_train_pipeline().get("user_selected_algorithm"),
            input_features=top_selected_features,
            model_update=False,
            job_timeout="1h",
            **params_dict,
        )
        db.session.commit()  # commit the task to db
        session["job_id"] = task.id  # add job id to session
    except Exception:
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

    response_object = {
        "status": "success",
        "data": {"task_id": task.id},
        "info": "Sit back and relax, Prometheus is training your model...",
    }
    return jsonify(response_object), 202


@bp.post("/set_algo_db/<int:project_id>")
@login_required
def set_algo_db(project_id: int) -> ResponseReturnValue:
    """Route to write the serialised models to the db"""
    task_id = request.get_json()["task_id"]  # get the task id from the client

    try:
        # Run two tasks (one for each model)
        _, job = current_user.launch_task(
            name="add_model_to_db",
            description="Update db with the tuned model",
            user_id=current_user.id,
            project_id=project_id,
            app_area="generic",
            task_id=task_id,
            model_type="tuned",
        )
        db.session.commit()  # commit the task to db

        task, _ = current_user.launch_task(
            name="add_model_to_db",
            description="Update db with the default model",
            user_id=current_user.id,
            project_id=project_id,
            app_area="generic",
            task_id=task_id,
            model_type="default",
            depends_on=job,
        )
        db.session.commit()  # commit the task to db
    except Exception:
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

    response_object = {
        "status": "success",
        "data": {"task_id": task.id},
        "info": "Sit back and relax, Prometheus is storing your models...",
    }
    return jsonify(response_object), 202


@bp.post("/set_algorithm_type/<int:project_id>")
@login_required
def set_algo_type(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the algorithm selected by the user
    selected_algorithm = request.get_json()["selected_algorithm"]

    # return an error when the user hasn't selected an algorithm
    if not selected_algorithm:
        return make_response(("Couldn't update the db, please try again!", 403))

    # update the db
    project.add_update_tmp_train({"user_selected_algorithm": selected_algorithm})
    response_dict = {
        "status": f"Successfully updated the algorithm: {selected_algorithm}",
        "pipeline": project.get_tmp_train_pipeline(),
    }

    return jsonify(response_dict)


@bp.post("/set_hyper/<int:project_id>")
@login_required
def set_hyper(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get the algorithm selected by the user
    selected_search_strategy = request.get_json()["selected_hyper"]

    # return an error when the user hasn't selected an algorithm
    if not selected_search_strategy:
        return make_response(("Couldn't update the db, please try again!", 403))

    # update the db
    project.add_update_tmp_train(
        {"user_selected_search_strategy": selected_search_strategy}
    )
    response_dict = {
        "status": f"Successfully updated the algorithm: {selected_search_strategy}",
        "pipeline": project.get_tmp_train_pipeline(),
    }

    return jsonify(response_dict)


@bp.post("/set_trained_model_params/<int:project_id>")
@login_required
def set_trained_model_params(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    pipe_tuned = pickle.loads(project.model_pipeline)

    # return an error when there is no existing model in the db
    if not project.model_pipeline:
        return make_response(("Couldn't update the db, please try again!", 403))

    if isinstance(pipe_tuned, Pipeline):
        trained_model_params = pipe_tuned.named_steps.get("algorithm").get_params()
    else:
        raise ValueError(f"Model of instance {type(pipe_tuned).__name__} not supported")

    project.add_update_tmp_train(
        {"user_selected_algorithm_params": trained_model_params}
    )

    response_dict = {
        "status": "Successfully updated the model",
        "pipeline": project.get_tmp_train_pipeline(),
    }

    return jsonify(response_dict)


@bp.route("/<int:project_id>/validate", methods=["POST"])
@login_required
def validate_model(project_id: int) -> ResponseReturnValue:
    """Run the background task of validating the models

    :param project_id: str
        The id of the current project
    """
    # Get the relevant project
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    try:
        task, _ = current_user.launch_task(
            name="model_validation",
            description="validation of model(s)",
            user_id=current_user.id,
            project_id=project_id,
            app_area=project.app_area,
            job_timeout=600,
        )
        db.session.commit()  # commit the task to db
        session["job_id"] = task.id  # add job id to session
    except Exception:
        current_app.logger.error("Unhandled exception", exc_info=sys.exc_info())

    response_object = {
        "status": "success",
        "data": {"task_id": task.id},
        "info": "Sit back and relax, Prometheus is validating your model...",
    }
    return jsonify(response_object), 202


@bp.post("/store_metrics/<int:project_id>")
@login_required
def store_metrics_pipeline(project_id: int) -> ResponseReturnValue:
    """Get the metrics in the right format from the validation task"""
    project = current_user.projects.filter_by(id=project_id).first_or_404()
    # get the results based on the task_id
    task_id = request.get_json()["task_id"]

    try:
        task = current_app.task_queue.fetch_job(task_id)
    except Exception:
        current_app.logger.error(
            f"Couldn't find the task with id {task_id}", exc_info=sys.exc_info()
        )

    metrics_dict = {"validation_metrics": task.result}
    project.add_update_tmp_train(metrics_dict)  # update the pipeline JSON with a dict
    response_dict = {
        "message": "Successfully added the performance metrics of the model(s)",
        "pipeline": project.get_tmp_train_pipeline(),
    }

    return jsonify(response_dict)


@bp.post("/set_metrics/<int:project_id>")
@login_required
def set_perform_metrics(project_id: int) -> ResponseReturnValue:
    """Write the validation metrics to the db"""
    project = current_user.projects.filter_by(
        id=project_id
    ).first_or_404()  # get the project from the db
    pipeline = (
        project.get_tmp_train_pipeline()
    )  # get the current train pipeline from the db
    pipeline_val_metrics = pipeline.get("validation_metrics")

    request_json = request.get_json()

    if project.app_area == "class_prediction":
        if request_json:
            decision_threshold = request.get_json()[
                "decision_threshold"
            ]  # get the decision threshold
        else:
            decision_threshold = 0.55  # set default value
    else:
        decision_threshold = "N/A"

    # select the metrics based on the threshold
    filtered_metrics_list = []
    filtered_image_list = []
    for key in pipeline_val_metrics.keys():
        filtered_thres_val_dict = pipeline_val_metrics.get(key)
        filtered_metrics_list.append(filtered_thres_val_dict.get("metrics"))
        filtered_image_list.append(filtered_thres_val_dict.get("images"))

    metrics_dict = {
        key: value
        for (key, value) in zip(
            list(pipeline_val_metrics.keys()), filtered_metrics_list
        )
    }
    images_dict = {
        key: value
        for (key, value) in zip(list(pipeline_val_metrics.keys()), filtered_image_list)
    }

    # update the db
    project.add_update_config(metrics_dict)
    project.add_update_config({"decision_threshold": decision_threshold})
    response_dict = {
        "message": "Successfully selected the performance metrics of the models",
        "metrics": metrics_dict,
        "decision_threshold": decision_threshold,
        "images": images_dict,
    }
    return jsonify(response_dict)


@bp.post("/projects/<int:project_id>/validation_results")
@login_required
def get_validation_metrics(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get tab id from user
    request_json = request.get_json()
    tab_id = request_json.get("tab_id")
    if tab_id == "tuned":
        metrics = project.project_config.get("tuned")
        print(
            f"The metrics of the model are: {project.project_config.get('tuned')}",
            file=sys.stderr,
        )
    elif tab_id == "default":
        metrics = project.project_config.get("default")
    else:
        raise ValueError(f"Validation tab not available for model {tab_id}")

    response_dict = {"metrics": metrics}

    return jsonify(response_dict)


@bp.post("/projects/<int:project_id>/launch")
@login_required
def launch_model(project_id: int) -> ResponseReturnValue:
    """Update the db with the temporary training pipeline data"""
    # We do that because maybe in the future we will save this params first in
    # server-side session (e.g. redis) before writing to the db
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # get selected model from the user
    request_json = request.get_json()
    selected_model = request_json.get("selected_model")

    # get the params from the temporary pipeline
    selected_target = project.get_tmp_train_pipeline().get("target_name")
    selected_features = project.get_tmp_train_pipeline().get("user_selected_features")
    selected_signals = project.get_tmp_train_pipeline().get("user_selected_signals")
    selected_transforms = project.get_tmp_train_pipeline().get(
        "user_selected_transforms"
    )
    selected_algorithm = project.get_tmp_train_pipeline().get("user_selected_algorithm")
    selected_decision_threshold = project.get_tmp_train_pipeline().get(
        "decision_threshold"
    )
    selected_features_after_ranking = project.get_tmp_train_pipeline().get(
        "user_selected_features_after_ranking"
    )

    # write it permanently to the db
    try:
        project.add_update_config({"target": selected_target})
        project.add_update_config({"user_selected_features": selected_features})
        project.add_update_config({"user_selected_signals": selected_signals})
        project.add_update_config({"user_selected_transforms": selected_transforms})
        project.add_update_config(
            {"user_selected_engineered_features": selected_features_after_ranking}
        )
        project.add_update_config({"user_selected_model": selected_model})
        project.update_algorithm(selected_algorithm)

        # write the decision threshold to the db only for classification
        if selected_decision_threshold:
            project.add_update_config(
                {"decision_threshold": selected_decision_threshold}
            )
    except Exception:
        # Log the error
        current_app.logger.exception("Could not launch the model. Problem with the db.")
        return make_response(("Not sure why, but we couldn't launch the model", 500))

    return jsonify(("Model launched successfully", 200))


# ================================================================================== #
#                                 INFERENCE FUNCTIONALITY                            #
# ================================================================================== #
@bp.route("/projects/<int:project_id>/inference")
@login_required
def run_inference(project_id: int) -> ResponseReturnValue:
    project = current_user.projects.filter_by(id=project_id).first_or_404()

    # Run the data processing task
    # TODO to be used in the progress bar
    task_process, job_process = current_user.launch_task(
        name="data_process",
        description="process-data",
        user_id=current_user.id,
        project_id=project_id,
        app_area=project.app_area,
        asset=project.asset,
        use_type="inference",
    )
    db.session.commit()  # commit the task to db

    # TODO: update the data processing task bar once this is done

    # get feature, signal, transforms lists from the db
    selected_features_list = project.project_config.get("user_selected_features")
    selected_transform_list = project.project_config.get("user_selected_transforms")

    # Run the feature extraction task
    task_extract_features, job_extract = current_user.launch_task(
        name="feature_extract_manual",
        description="manual feature extraction",
        user_id=current_user.id,
        project_id=project_id,
        app_area=project.app_area,
        asset=project.asset,
        use_type="inference",
        features=selected_features_list,
        transforms=selected_transform_list,
        job_timeout="1h",
        depends_on=job_process,
    )
    db.session.commit()  # commit the task to db

    # TODO update the feature extraction task bar once this is done

    # Run the inference task
    task, _ = current_user.launch_task(
        name="model_inference",
        description="run inference on data",
        user_id=current_user.id,
        project_id=project.id,
        app_area=project.app_area,
        depends_on=[job_process, job_extract],
    )
    db.session.commit()  # commit the task to db
    session["job_id"] = task.id  # add job id to session

    response_object = {
        "status": "success",
        "data": {"task_id": task.id},
        "info": "Sit back and relax, Prometheus is working on your predictions...",
    }
    return jsonify(response_object), 202
