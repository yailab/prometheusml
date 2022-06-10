from flask import Blueprint

# the name of the blueprint and the name of the base module
bp = Blueprint("errors", __name__)

# import at the end to avoid circular dependencies
from webapp.errors import handlers  # noqa: F401, E402
