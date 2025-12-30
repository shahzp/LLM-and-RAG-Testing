import re


def test_logical_consistency(llm_client):
    """
    Checks how consistent response from LLM is to equivalent questions
    :param llm_client:
    :return:
    """
    equivalent_questions=['what is the capital of Japan?','Tokyo is capital of which country?']
    responses=[]
    words=[]
    for question in equivalent_questions:
        response=llm_client.generate(question)
        responses.append(re.sub(r'[^a-z\s]', '', response['text'].lower()))

    for response in responses:
        split_res=set(response.split(' '))
        words.append(split_res)
    if words:
        overlap=words[0].intersection(words[1])
        union=words[0].union(words[1])
        overlap_ratio=len(overlap)/len(union)
        print(overlap)
        print(responses)
        assert overlap_ratio>=0.4, 'Answers not consistent for equivalent questions'


def test_error_handling(llm_client):
    """
    Checks model response when question is incorrect
    :param llm_client:
    :return:
    """
    incorrect_prompts=['What is the color of silence?','How many corners does circle have?']
    indicator_words = ["does not",
        "doesn't",
        "no",
        "cannot",
        "undefined",
        "abstract",
        "not possible",
        "does not exist",
        "zero"]
    for prompt in incorrect_prompts:
        response=llm_client.generate(prompt)
        print(response['text'])
        assert len(response['text'].strip()) >=10, f'Response too short for incorrect input {prompt}'
        indicator_check=any([indicator in response['text'] for indicator in indicator_words])
        assert indicator_check, f'indicators not present in incorrect input {prompt}'




