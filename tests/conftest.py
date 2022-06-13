import numpy as np
import pytest

from webapp import create_app, db
from webapp.models import Role, User


@pytest.fixture()
def arr_one():
    np.random.seed(141)
    return np.random.rand(
        100,
    )


@pytest.fixture()
def arr_two():
    np.random.seed(132)
    return np.random.rand(
        100,
    )


@pytest.fixture(scope="module")
def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture(scope="module")
def new_user_role():
    user_role = Role(name="user")
    return user_role


@pytest.fixture(scope="module")
def new_user(app):
    with app.app_context():
        user = User(
            username="test_user",
            email="email@example.com",
            register_newsletter=True,
            register_discord=False,
        )
        user.set_password("test_pass1!*")
        user.add_role("user")
    return user
