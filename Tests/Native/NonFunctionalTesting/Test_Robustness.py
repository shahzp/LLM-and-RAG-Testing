def test_input_variations(llm_client):
    """
    Checks if model response is accurate for different variations
    :param llm_client:
    :return:
    """
    variations=['what is the capital of france?','What is the capital of France?','capital of france?']
    for variation in variations:
        response=llm_client.generate(variation)
        print(f'response is {response["text"]}')
        assert 'paris' in response['text'].lower(),'response does not contain word paris'

