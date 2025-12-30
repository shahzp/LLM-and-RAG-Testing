import json
import re
import pytest

@pytest.mark.functional
def test_list_format(llm_client):
    """
    To check if LLM can format text as a list of strings
    :param llm_client:
    :return: None
    """
    prompt='list 3 fruits. Format your answer in a numbered list'
    response=llm_client.generate(prompt=prompt)
    pattern=r'[1-3][.)]'
    matches=re.findall(pattern,response['text'])
    assert 2 <= len(matches) <= 4,'Response is not numbered list'
    print(f'response is {response["text"]}')

def test_simplejson(llm_client):
    """
    To check if LLM can generate json with given values
    :param llm_client:
    :return: None
    """
    prompt="create a json with a person's name and age. Use the name Sastry and age 30 "
    response=llm_client.generate(prompt=prompt)
    json_text=response['text'].strip()
    try:
        data=json.loads(json_text)
    except json.JSONDecodeError as e:
        pytest.fail(f'Error parsing json: {e}')

    assert data.get("name","").lower()=='sastry','name is missing in json'
    assert data.get("age","")==30,'age is missing in json'
    print(f'{response["text"]}')
