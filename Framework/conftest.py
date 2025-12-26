import pytest
from deepeval.models import GPTModel

from Framework.OpenAITestClient import OpenAITestCleint

@pytest.fixture(scope='session')
def base_model():
    return {
        'model': 'gpt-4o'
    }

@pytest.fixture()
def evaluation_model_factory(base_model):
    def _create(temperature=0.0):
        return GPTModel(model=base_model['model'],temperature=temperature)

    return _create

@pytest.fixture(scope='session')
def client_model():
    client= OpenAITestCleint()
    if client:
        return client
    else:
        pytest.skip('Model under test is not available')