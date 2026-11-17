import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from app import create_app

@pytest.fixture()
def app(tmp_path):
    app = create_app(tmp_path/'data')
    app.config['TESTING'] = True
    return app

@pytest.fixture()
def client(app):
    return app.test_client()

@pytest.fixture()
def headers(app):
    return {'X-VV-Token':app.extensions['csrf']}
