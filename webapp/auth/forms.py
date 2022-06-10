from flask_wtf import FlaskForm, RecaptchaField
from wtforms import BooleanField, PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

from webapp.models import User


# include the classes to represent the web forms
class LoginForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter username"},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter password"},
    )
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Log In")


class RegistrationForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired()],
        render_kw={"placeholder": "...or how PrometheusML should address you"},
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()],
        render_kw={"placeholder": "...to make sure you are human and not AI"},
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8, message="The password should be at least 8 characters long"),
        ],
        render_kw={"placeholder": "...type it here. Shhh don't tell anyone"},
    )
    password2 = PasswordField(
        "Repeat Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match"),
        ],
        render_kw={"placeholder": "...practice makes perfect"},
    )
    recaptcha = RecaptchaField()
    submit = SubmitField("Register")

    newsletter = BooleanField("Subscribe to our newsletter", default=True)
    discord = BooleanField("Join our Discord community", default=True)

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError("Please use a different username.")

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError("Please use a different email address.")
