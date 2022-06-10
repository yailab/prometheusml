import logging
import os
from logging.handlers import RotatingFileHandler

import rq
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_bootstrap import Bootstrap
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from redis import Redis
from rq.registry import FailedJobRegistry, StartedJobRegistry

from config import Config

# create instances of the various flask extensions
db = SQLAlchemy()
migrate = Migrate()
bootstrap = Bootstrap()
bcrypt = Bcrypt()
csrf = CSRFProtect()
login = LoginManager()
login.login_view = "auth.login"
login.refresh_view = "auth.login"
login.needs_refresh_message = "Session timed-out, please re-login"
login.needs_refresh_message_category = "info"


def register_blueprints(app):
    """
    Register the various blueprints for the application.
    """
    from webapp.errors import bp as errors_bp

    app.register_blueprint(errors_bp)

    from webapp.auth import bp as auth_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")

    from webapp.main import bp as main_bp

    app.register_blueprint(main_bp)


def create_app(config_class=Config):
    """
    The `application factory` function to create the `Flask` instance. This is done for simpler testing,
    and to run different version of the same application (multiple instances).

    :parameter config_class:
               A configuration class for the application
    :returns Flask application instance
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialise the flask extensions
    db.init_app(app)
    migrate.init_app(app, db)
    bootstrap.init_app(app)
    bcrypt.init_app(app)
    csrf.init_app(app)
    login.init_app(app)

    # configure redis and task queue
    app.redis = Redis.from_url(app.config["REDIS_URL"])
    # the task queue is where the tasks are submitted
    app.task_queue = rq.Queue("prometheusML-tasks", connection=app.redis)
    # the registry for RQ failed jobs -- along with their exception information (type, value, traceback).
    app.failed_registry = FailedJobRegistry(queue=app.task_queue)
    app.started_registry = StartedJobRegistry(queue=app.task_queue)

    # Register the various blueprints (app subsets) to associate them with the application
    register_blueprints(app)

    # logging setup only when flask in dev mode
    if not app.debug and not app.testing:
        # configuration to output errors in a log file
        if not os.path.exists("logs"):
            os.mkdir("logs")
        file_handler = RotatingFileHandler(
            "logs/prometheusML.log", maxBytes=10240, backupCount=10
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s " "[in %(pathname)s:%(lineno)d]"
            )
        )
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info("yaiLab startup")

    return app


# We load them at the end of the file to avoid `circular imports`
from webapp import models, utils  # noqa: F401, E402
