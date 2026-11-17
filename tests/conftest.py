import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app import create_app


class FakeProvider:
    def __init__(self):
        self.calls = []
    def models(self):
        return {'online':True,'models':[{'name':'test:local','size':10000000}]}
    def validate(self, name):
        if name != 'test:local': raise ValueError('Unknown local model')
    def stream(self, model, messages, cancelled):
        self.calls.append(messages)
        yield {'type':'token','text':'A local answer [1].'}
        yield {'type':'metrics','tokens':7,'tokens_per_second':12,'reason':'stop'}


@pytest.fixture()
def app(tmp_path):
    app = create_app(tmp_path/'data', FakeProvider())
    app.config['TESTING'] = True
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def headers(app):
    return {'X-VV-Token':app.extensions['csrf']}
