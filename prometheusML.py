from webapp import cli, create_app, db

# import the various models
from webapp.models import Notification, Project, Role, Task, User

# Create an instance of the application
app = create_app()
cli.register(app)


# register the function as a shell context function for `flask shell` command
# that is done to test things fast in a Python shell
@app.shell_context_processor
def make_shell_context():
    """
    Pre-import the db instance and the models to the shell session.
    """
    ret_dict = {
        "db": db,
        "User": User,
        "Role": Role,
        "Project": Project,
        "Task": Task,
        "Notification": Notification,
    }
    return ret_dict
