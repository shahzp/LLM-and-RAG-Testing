import pytest


@pytest.mark.functional
def test_basic_responses(llm_client):
    """
    Checks if LLM is providing proper response with minimum length
    :param llm_client:
    :return: None
    """
    response = llm_client.generate('Hello, How are you?')
    assert isinstance(response, dict), "Response is not a dictionary"
    assert response['text'].strip() !=''
    assert len(response['text']) > 10, 'response too short'
    print(f'basic response test: response length is {len(response["text"])}')

@pytest.mark.functional
def test_follow_instructions(llm_client):
    """
    Checks if LLM is following instructions correctly
    :param llm_client:
    :return: None
    """
    #Assuming these are common colors
    colors=['red','orange','yellow','green','blue','purple','pink',
            'brown','black','white','gray','cyan','magenta','gold','silver','maroon','navy','olive']
    response=llm_client.generate('Name exactly three common colors. Just list them')
    assert isinstance(response, dict), "Response is not a dictionary"
    count=sum([1 for color in colors if color in response['text'].lower()])
    assert count>=2,'Expected atleast two colors.'
    print(f'color count in response: {count}')
    print(response['text'])

@pytest.mark.functional
def test_simple_qa(llm_client):
    """
    Checks basic facts.
    :param llm_client:
    :return: None
    """
    response=llm_client.generate('What is the capital of France?')
    assert isinstance(response, dict), "Response is not a dictionary"
    assert 'paris' in response['text'].lower(), 'Response does not contain paris.'
    print(f'{response["text"]}')

@pytest.mark.functional
def test_multi_turn_basic(llm_client):
    """
    Checks if LLM model can retain the context. This is a basic check
    :param llm_client:
    :return: None
    """
    prompt="""
    user:My name is Sam
    Assistant: Hi Sam, How may I help you?
    User: What is my name?
    """
    response=llm_client.generate(prompt)
    print(f'response is {response["text"]}')
    assert isinstance(response, dict), "Response is not a dictionary"
    assert 'sam' in response['text'].lower(), 'Name not identified correctly'