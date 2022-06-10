import pickle
import sys

from prometheus.visualisation.data_profiling import profile_dataset
from webapp import create_app, db
from webapp.models import User
from webapp.utils_task import get_uploaded_file_paths, read_uploaded_files

app = create_app()
app.app_context().push()


def data_profile(user_id: int, project_id: int, selected_features: list) -> str:
    """Create the HTML report for data profiling"""
    # get the paths of the uploaded dataset(s)
    uploaded_file_paths = get_uploaded_file_paths(user_id, project_id, "training")

    # read the uploaded dataset(s)
    # select only the first one for now
    # TODO: support data profiling with multiple datasets
    df = read_uploaded_files(
        uploaded_file_paths, use_type="training", use_cols=selected_features
    )[0]

    # subset based on the features & generate the report
    html_report = profile_dataset(df[selected_features])

    return html_report


def add_model_to_db(
    user_id: int, project_id: int, task_id: str, model_type: str = "tuned"
) -> None:
    """Add a sklearn pipeline to db"""
    # Get the user and the project
    user = User.query.filter_by(id=user_id).first()
    project = user.projects.filter_by(id=project_id).first()

    task = app.task_queue.fetch_job(task_id)  # fetch the job from redis
    (
        tuned_pipe,
        default_pipe,
    ) = task.result  # get the results of the model  // TODO: for memory have it here

    if model_type not in ["tuned", "default"]:
        raise ValueError(f"Model type {model_type} not currently supported!")

    if model_type == "tuned":
        pipeline = tuned_pipe
    else:
        pipeline = default_pipe

    # pickle the models with protocol 5 for large data buffers:
    # https://joblib.readthedocs.io/en/latest/persistence.html
    pickled_pipeline = pickle.dumps(pipeline, protocol=5)
    try:
        if model_type == "tuned":
            project.model_pipeline = pickled_pipeline  # add it to the project db
        else:
            project.model_pipeline_default = (
                pickled_pipeline  # add it to the project db
            )
        db.session.commit()
    except BaseException:
        app.logger.error("Unhandled exception", exc_info=sys.exc_info())
        raise
