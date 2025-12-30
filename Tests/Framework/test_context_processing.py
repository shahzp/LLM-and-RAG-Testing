from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def test_list_format(client_model,evaluation_model_factory):
    prompt="List exactly 3 fruits.Format your answer as a numbered list"
    response=client_model.generateText(prompt)
    response_text=response['text']
    evaluation_model=evaluation_model_factory(temperature=0.0)
    testcase=LLMTestCase(input=prompt,actual_output=response_text)

    list_metric=GEval(name='list format compliance',
          criteria="""
          Evaluate if the response properly follows the instruction requesting to follow list format.
          The response should contain exactly 3 fruits and the fruit names should be recognizable common names.
          Response should follow proper formatting between lines.
          Give full score only if all the above mentioned requirements are met.
          """,model=evaluation_model,threshold=0.8,
          evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT])

    assert_test(testcase,[list_metric])


def test_simple_json(client_model,evaluation_model_factory):
    prompt="Create a well formatted json with person's name and age. Use the name Sastry and age 25"
    response=client_model.generateText(prompt)
    response_text=response['text']
    evaluation_model=evaluation_model_factory(temperature=0.0)
    testcase=LLMTestCase(input=prompt,actual_output=response_text)
    json_metric=GEval(name='simple json compliance',
                      criteria="""
                      Evaluate if the response meets the following requirements.
                      1.The response is a proper json format with { } and double quotes where required
                      2. There should be key value pairs with keys as name and age
                      3. Value for name should be sastry or Sastry and value for age should be 25
                      4.The json should be parsable and well formatted
                      
                      Deduct points for invalid json syntax and missing fields and values.
                      """,model=evaluation_model,threshold=0.8,
                      evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT])
    assert_test(testcase,[json_metric])