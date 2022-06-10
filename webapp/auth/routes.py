from flask import Markup, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user
from werkzeug.urls import url_parse

from webapp import db
from webapp.auth import bp
from webapp.auth.forms import LoginForm, RegistrationForm
from webapp.models import User


@bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Implements the login feature for the app.
    Errors are shown if incorrect details are used. If the user tried
    to access a page requiring login without being authenticated,
    they are redirected there after sign in.
    """
    if current_user.is_authenticated and current_user.has_role("user"):
        return redirect(url_for("main.index"))
    elif current_user.is_authenticated and current_user.has_role("admin"):
        return redirect(url_for("admin.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash(Markup("Invalid username or password"), category="error")
            return redirect(url_for("auth.login"))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get("next")
        """To prevent malicious users from adding a malicious site into the parameters,
        this checks to see if the url is relative.
        """
        if not next_page or url_parse(next_page).netloc != "" or next_page == "/logout":
            if user.has_role("user"):
                next_page = url_for("main.index")
            elif user.has_role("admin"):
                next_page = url_for("admin.index")
        return redirect(next_page)

    return render_template("auth/login.html", title="Sign In", form=form)


@bp.route("/register", methods=["GET", "POST"])
def register():
    """Implements the registration feature for the app."""
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            register_newsletter=form.newsletter.data,
            register_discord=form.discord.data,
        )
        user.set_password(form.password.data)
        user.add_role("user")

        try:
            db.session.add(user)

            flash(
                Markup("Congratulations, you are now a registered user!"),
                category="success",
            )
        except BaseException:
            db.session.rollback()
            flash(Markup("Could not add the user (contact yaiLab)!"), category="error")
            raise

        try:
            user.add_project(
                name="General-regression-problem",
                asset="other",
                app_area="continuous_prediction",
            )
            user.add_project(
                name="General-classification-problem",
                asset="other",
                app_area="class_prediction",
            )
        except BaseException:
            db.session.rollback()
            flash(
                Markup("Could not add the projects (contact yaiLab)!"), category="error"
            )
            raise

        db.session.commit()

        return redirect(url_for("main.index"))
    return render_template("auth/register.html", title="Register", form=form)


@bp.route("/terms")
def get_terms():
    return render_template("auth/terms.html", title="Terms and conditions")


@bp.route("/data_privacy")
def get_data_policy():
    return render_template("auth/data_privacy.html", title="Data policy")


@bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("main.index"))
