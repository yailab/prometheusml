from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import BooleanField, DecimalField, RadioField, SubmitField
from wtforms.validators import DataRequired


class FileUploadForm(FlaskForm):
    photo = FileField("File", validators=[FileRequired()])
    submit = SubmitField("Submit")


class InputDataForm(FlaskForm):
    npratio = DecimalField(
        "N/P ratio",
        validators=[DataRequired()],
        render_kw={"placeholder": "0.0", "min": "0.0"},
    )
    voltage_range = DecimalField(
        "Discharge cut-off voltage",
        validators=[DataRequired()],
        render_kw={"placeholder": "0.0", "min": "0.0"},
    )
    submit = SubmitField("Submit values")


class ModelDataForm(FlaskForm):
    # for the checkbox inputs
    voltage_feature = BooleanField("voltage_feature")
    temp_feature = BooleanField("temp_feature")
    # radio items
    app_predict = RadioField("app_predict")
