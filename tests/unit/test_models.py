"""Module of unit tests for the database models"""
from webapp.models import Role


def test_new_user_role(new_user_role):
    """
    GIVEN a Role model
    WHEN a new 'user' Role is created
    THEN check the name field(s) are defined correctly
    """
    assert new_user_role.name == "user"


def test_new_user(new_user):
    """
    GIVEN a User model
    WHEN a new User is created
    THEN check the username, email, hashed_password, role, newsletter and discord subscription
        fields are defined correctly
    """
    assert new_user.username == "test_user"
    assert new_user.email == "email@example.com"
    assert new_user.password_hash != "test_pass1!*"
    assert Role(name="user") in new_user.roles
    assert new_user.register_newsletter
    assert not new_user.register_discord
