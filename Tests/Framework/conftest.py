import os

import pytest
from deepeval.models import GPTModel

from RAG.ShoeStoreRAG import ShoeStoreRAG
from Tests.Framework.OpenAITestClient import OpenAITestCleint

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

@pytest.fixture(scope='function')
def setup_rag_tests():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    openai_api_key=os.getenv('OPENAI_API_KEY')
    shoe_rag=ShoeStoreRAG(pinecone_api_key=pinecone_api_key,openai_api_key=openai_api_key)
    return shoe_rag

