
def test_hallucination_detection(llm_client):
    """
    Checking basic hallucination detection
    :param llm_client:
    :return:
    """
    impossible_questions=['what is the name of the president of wakanda?',
                          'what is the color of favourite mobile of Napolean Bonaparte?']
    notExpected_words=['the president of wakanda is','color of favourite mobile Napolean Bonaparte is',
                       'the name of the president is','color is','']
    for question in impossible_questions:
        response=llm_client.generate(question)
        is_hallucinated=not any([word in response['text'].lower() for word in notExpected_words])
        print(f'response is {response["text"]}')
        assert is_hallucinated,f'Model may be hallucinating for {question}'