from flask import render_template

from webapp import db

# Import the errors blueprint
from webapp.errors import bp


@bp.app_errorhandler(404)
def not_found_error(error):
    return render_template("errors/404.html"), 404


@bp.app_errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template("errors/500.html"), 500


# @bp.app_errorhandler(413)
# def too_large(error):
#     return "File is too large", 413
