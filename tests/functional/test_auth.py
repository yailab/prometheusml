"""Module of functional tests for the authorisation routes"""
from flask import url_for


def test_login_page(client):
    assert client.get(url_for("auth.login")).status_code == 200


def test_register_page(client):
    assert client.get(url_for("auth.register")).status_code == 200
