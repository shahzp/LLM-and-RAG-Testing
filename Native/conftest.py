import  pytest

from Llamatesting import LlamaTestClient

@pytest.fixture
def llm_client(scope='session'):
    llm=LlamaTestClient('tinyllama', host='localhost',port=11434)
    isAvailable=llm.model_availability()
    if not isAvailable:
        pytest.skip("Model Not Available. Please run 'ollama pull tinyllama' first")
    return llm
