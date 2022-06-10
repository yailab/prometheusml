from flask import Blueprint

bp = Blueprint("main", __name__)

# import at the end to avoid circular dependencies
from webapp.main import routes  # noqa: F401, E402
