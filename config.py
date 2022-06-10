import os

basedir = os.path.abspath(os.path.dirname(__file__))


#  Need to configure Flask and the flask extensions
class Config(object):
    # secret key to be used as a cryptographic key, useful to generate signatures or tokens
    # in production server the `secret key` should be something that nobody else knows
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess"

    # Protect against XSS
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True

    # Use sqlite database as a fallback value
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL"
    ) or "sqlite:///" + os.path.join(basedir, "app.db")
    # disable the signal to the application every time a change is about to be made in the database.
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # eliminate database errors due to stale pooled connections
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # Add redis server URL
    REDIS_URL = os.environ.get("REDIS_URL") or "redis://"

    # reCaptcha
    RECAPTCHA_PUBLIC_KEY = (
        os.environ.get("RC_SITE_KEY") or "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"
    )
    RECAPTCHA_PRIVATE_KEY = (
        os.environ.get("RC_SECRET_KEY") or "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe"
    )

    # Upload file config
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    ALLOWED_EXTENSIONS = [".csv"]
    UPLOAD_FOLDER = "uploads"
