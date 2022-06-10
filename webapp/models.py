import json
from datetime import datetime
from hashlib import md5

import redis
import rq
from flask import current_app
from flask_login import UserMixin

from webapp import bcrypt, db, login
from webapp.utils import NumpyEncoder

# Helper table for the many-to-many relationship of User and Role
roles_users = db.Table(
    "roles_users",
    db.Column("user_id", db.Integer(), db.ForeignKey("user.id")),
    db.Column("role_id", db.Integer(), db.ForeignKey("role.id")),
)


class User(UserMixin, db.Model):
    """
    Class for creating the db schema of the User.
    """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128), nullable=False)
    register_timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    register_newsletter = db.Column(db.Boolean(), nullable=False)
    register_discord = db.Column(db.Boolean(), nullable=False)
    active = db.Column(db.Boolean())
    # database view for one-to-many relationship with projects
    projects = db.relationship("Project", backref="user", lazy="dynamic")
    # relationship with the notifications db
    notifications = db.relationship("Notification", backref="user", lazy="dynamic")
    # create a table to maintain state around running tasks
    tasks = db.relationship("Task", backref="user", lazy="dynamic")
    # db view for many-to-many relationship between users and roles
    roles = db.relationship(
        "Role", secondary=roles_users, backref=db.backref("users", lazy="dynamic")
    )

    def __repr__(self):
        """
        Overload the printing operator
        :return:
        """
        return "<User {}>".format(self.username)

    def __unicode__(self):
        """
        Overload the print method. For the administrative interface.
        """
        return self.username

    def set_password(self, password):
        """
        Method for hashing the password
        :param password:
        :return: hashed password
        """
        # Decode the byte strings to UTF-8 strings (to add them in PostgresSQL db)
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        """
        Method to check the hashed password.

        :param password:
        :return: boolean whether or not the password is the same
        """
        # Encode the unicode password string from the db into bytes string
        bytes_password_hash = self.password_hash.encode("utf-8")
        return bcrypt.check_password_hash(bytes_password_hash, password)

    def has_role(self, role):
        """
        Method to check if user has a specified role.
        :param role: A role name or `Role` instance
        :return: `True` if the user identifies with the specified role.
        """
        if isinstance(role, str):
            return role in (role.name for role in self.roles)
        else:
            return role in self.roles

    def avatar(self, size):
        """
        Method to create an avatar based on gravatar.

        :param size: size of gravatar resolution
        :return: avatar picture
        """
        digest = md5(self.email.lower().encode("utf-8")).hexdigest()
        return "https://www.gravatar.com/avatar/{}?d=identicon&s={}".format(
            digest, size
        )

    def add_project(self, name, **kwargs):
        """
        Method to add a new project to a user. If project with the same name
        exists raise an exception?
        :param name: the name of the project to create
        """
        project = Project(project_name=name, user=self, **kwargs)
        if project not in self.projects.all():
            db.session.add(project)
            return True
        return False

    def delete_project(self, name):
        """
        Method to delete the project of a user.
        """
        rv = False
        if self.projects.filter_by(project_name=name).first():
            rv = True
            self.projects.filter_by(project_name=name).delete()
        return rv

    def add_role(self, role: str) -> bool:
        """
        Adds a role to a user.
        :param role: The role to be added for the user.
        """
        role = Role.query.filter_by(name=role).first_or_404()
        if role not in self.roles:
            self.roles.append(role)
            db.session.add(self)
            return True
        return False

    def remove_role(self, role: str) -> bool:
        """
        Removes a role from a user.
        :param role: The role to remove from the user
        """
        rv = False
        role = Role.query.filter_by(name=role).first_or_404()
        if role in self.roles:
            rv = True
            self.roles.remove(role)
            db.session.add(self)
        return rv

    def add_notification(self, name, data):
        """
        A helper method to add a notification for the user to the db, and also
        remove it, if it has the same name.
        :param name: name of the notification
        :param data: data of the notification
        :return:
        """
        self.notifications.filter_by(name=name).delete()
        n = Notification(name=name, payload_json=json.dumps(data), user=self)
        db.session.add(n)
        return n

    def launch_task(self, name: str, description: str, app_area: str, *args, **kwargs):
        """
        Helper method submitting a task to the RQ queue,and add it to the db.
        It adds the new task object to the session.

        :param name: function name of the task
        :param description: description of the task
        :param app_area:
        :param args: positional arguments to be passed in the task
        :param kwargs: keyword arguments to be passed in the task
        :return: the task
        """
        # current_app.task_queue to access it anywhere in the app

        # listen to the respective module for the respective application area
        func_path = "webapp.tasks_" + app_area + "."

        rq_job = current_app.task_queue.enqueue(func_path + name, *args, **kwargs)

        task = Task(id=rq_job.get_id(), name=name, description=description, user=self)

        # Check this link for dependencies:
        # https://python-rq.org/docs/#job-dependencies

        # write the task info to the database
        db.session.add(task)
        return task, rq_job

    def get_tasks_in_progress(self):
        """
        Helper method that returns the complete list of functions that are outstanding for the user.
        :return:
        """
        return Task.query.filter_by(user=self, complete=False).all()

    def get_task_in_progress(self, name):
        """
        Helper method that returns a specific task.
        :param name: function name of the task
        :return:
        """
        return Task.query.filter_by(name=name, user=self, complete=False).first()


class RoleMixin(object):
    """Mixin for `Role` model definitions"""

    def __eq__(self, other):
        return self.name == other or self.name == getattr(other, "name", None)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)


class Role(db.Model, RoleMixin):
    """
    Class for creating the db schema of the users' Roles.
    """

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(128))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Project(db.Model):
    """
    Class for creating the db schema of the Project.
    """

    id = db.Column(db.Integer, primary_key=True)
    project_name = db.Column(db.String(140))
    asset = db.Column(db.String(140))
    app_area = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    algorithm = db.Column(db.String(140), nullable=True)
    model_pipeline = db.Column(db.LargeBinary, nullable=True)
    model_pipeline_default = db.Column(db.LargeBinary, nullable=True)

    # Project-specific attributes (e.g. capacity_threshold) in a json format -- This column's entries can be empty.
    # Be careful how the changes are detected with SQLAlchemy. Check the following blogpost:
    # https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/
    project_config = db.Column(db.JSON, nullable=True)
    tmp_train_pipeline = db.Column(db.JSON, nullable=True)
    # connect the project with the user
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    def __repr__(self):
        return "<Project for {} predicting {} using {}>".format(
            self.asset, self.app_area, self.algorithm
        )

    def update_algorithm(self, algorithm):
        self.algorithm = algorithm
        db.session.commit()

    def add_update_tmp_train(self, data_dict: dict):
        """
        A helper method to add/append/update data to the column for the temporary training pipeline.
        """
        # TODO: add support for multiple models
        # default structure -- to be able to show the tree structure
        if not self.tmp_train_pipeline:
            init_pipeline_dict = {
                "user_selected_features": None,
                "target_name": None,
                "user_selected_signals": None,
                "user_selected_transforms": None,
                "user_selected_features_after_ranking": None,
                "user_selected_algorithm": None,
                "user_selected_algorithm_params": None,
            }
            self.tmp_train_pipeline = json.dumps(init_pipeline_dict)
            db.session.add(self)
            db.session.commit()

        # encode numpy arrays to int, float before pushing to the db
        updated_dict = {**self.get_tmp_train_pipeline(), **data_dict}
        updated_json = json.dumps(updated_dict, cls=NumpyEncoder)

        # Update the current project train pipeline in the db
        self.tmp_train_pipeline = updated_json
        db.session.add(self)
        db.session.commit()

    def get_tmp_train_pipeline(self):
        return json.loads(self.tmp_train_pipeline)

    def add_update_config(self, data_dict: dict):
        """
        A helper method to add/append/update project-specific config data. It updates the value if the config key
        already exists, otherwise it appends them to the json.
        :param data_dict: config metadata as a python dictionary
        :return:
        """
        if not self.project_config:
            self.project_config = {}
        # Update the current project config dict
        self.project_config = {**self.project_config, **data_dict}
        db.session.commit()

    def remove_config_data(self):
        """
        A helper method to remove project-specific metadata.
        :return:
        """
        pass


class Notification(db.Model):
    """
    Class for creating the db schema of the Notification.
    A notification will have a name, an associated user, an associated task,
    a timestamp, and a payload
    """

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    # need to add the related task for the notification
    task_id = db.Column(db.String(36), db.ForeignKey("task.id"))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    payload_json = db.Column(db.Text)

    def get_data(self):
        return json.loads(str(self.payload_json))


class Task(db.Model):
    """
    Class for the various tasks run by the users.
    """

    # as an ID use the job identifiers generated by RQ
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(128), index=True)
    description = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    # status = db.Column(db.String(36))
    # complete = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return "<Task {} for user: {}, with id: {}>".format(
            self.name, self.user_id, self.id
        )

    def get_rq_job(self):
        """
        Helper method that loads the RQ `Job` instance from a given task id.

        :return: an RQ `Job` object
        """
        try:
            # get the `Job` instance given a task id
            rq_job = rq.job.Job.fetch(self.id, connection=current_app.redis)
        except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
            return None
        return rq_job

    def get_progress(self):
        """
        Method to get the progress of a task by reading the meta attribute `progress`.
        :return: float: task progress
        """
        job = self.get_rq_job()
        return job.meta.get("progress", 0) if job is not None else 100


# load a user from a db given its ID -- flask-login
@login.user_loader
def load_user(uid):
    return User.query.get(int(uid))
