from flask import Blueprint

# the name of the blueprint and the name of the base module
bp = Blueprint("auth", __name__)

# import at the end to avoid circular dependencies
from webapp.auth import routes  # noqa: F401, E402
