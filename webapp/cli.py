import click
from redis import Redis
from rq import Connection, Worker

from webapp import db
from webapp.models import Project, Role, User


def register(app):
    @app.cli.command("run_worker")
    def run_worker():
        redis_url = app.config["REDIS_URL"]
        app.redis_connection = Redis.from_url(redis_url)
        with Connection(app.redis_connection):
            worker = Worker("prometheusML-tasks")
            worker.work()

    @app.cli.command("create_role")
    @click.argument("name")
    @click.argument("description")
    def create_role(name, description):
        role = Role(name=name, description=description)
        db.session.add(role)
        db.session.commit()

    @app.cli.command("create_user")
    @click.argument("username")
    @click.argument("password")
    @click.argument("email")
    @click.argument("role")
    def create_user(username, password, email, role):
        user = User(username=username, email=email)
        user.set_password(password)
        role = Role.query.filter_by(name=role).first()
        user.add_role(role)
        db.session.add(user)
        db.session.commit()

    @app.cli.command("create_project")
    @click.argument("username")
    @click.argument("project_name")
    @click.argument("asset")
    @click.argument("app_area")
    @click.argument("algorithm")
    def create_project(username, project_name, asset, app_area, algorithm):
        try:
            user = User.query.filter_by(username=username).first_or_404()
            project = Project(
                project_name=project_name,
                asset=asset,
                app_area=app_area,
                algorithm=algorithm,
                user=user,
            )
            db.session.add(project)
            db.session.commit()
        except BaseException:
            pass

    @app.cli.command("load_roles_db")
    def load_roles():
        if not Role.query.filter_by(name="user").first():
            user_role = Role(name="user")
            db.session.add(user_role)
            db.session.commit()

        if not Role.query.filter_by(name="admin").first():
            admin_role = Role(name="admin")
            db.session.add(admin_role)
            db.session.commit()
