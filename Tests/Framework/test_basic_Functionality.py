from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def test_basic_response(client_model,evaluation_model_factory):
    evaluation_model=evaluation_model_factory(temperature=0.0) #Judge
    prompt='Hi,How are you?'
    response=client_model.generateText(prompt)
    response_text=response['text']

    testcase=LLMTestCase(input=prompt,actual_output=response_text)

    response_quality_metric=GEval(name='response_quality',
                                  criteria="""
                                  Evaluate whether the response is an appropriate greeting.
                                  The response should:
                                  - Acknowledge the greeting
                                  - Maintain a friendly tone
                                  - Avoid irrelevant or factual content
                                  """ ,
          evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT],
          threshold=0.7,
          model=evaluation_model
          )
    assert len(response_text.strip()) > 10, 'Response too short'
    assert_test(testcase,[response_quality_metric])
    #Log judge reasoning and score
    print(f'Score is {response_quality_metric.score}')
    print(f'reasoning is {response_quality_metric.reason}')

def test_instruction_following(client_model,evaluation_model_factory):
    evaluation_model = evaluation_model_factory(temperature=0.0)  # Judge
    prompt = 'Name three colors. Just list the colors'
    response = client_model.generateText(prompt)
    response_text = response['text']

    testcase=LLMTestCase(input=prompt,actual_output=response_text)
    response_metric=GEval(name='following_instruction',
                         criteria="""
                         Determine whether the response follows the instruction.
                         The response must list exactly three common colors
                         as a simple list without explanations.
                         """,
                          evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT],
                          threshold=0.7,
                          model=evaluation_model)
    assert_test(testcase,[response_metric])
    print(f'Score is {response_metric.score}')
    print(f'reasoning is {response_metric.reason}')

def test_simple_qa(client_model,evaluation_model_factory):
    evaluation_model = evaluation_model_factory(temperature=0.0)
    prompt="What is the capital of France?"
    response = client_model.generateText(prompt)
    response_text = response['text']
    testcase=LLMTestCase(input=prompt,actual_output=response_text)
    factual_metric=GEval(name='simple_qa',criteria="""
    Determine whether the model correctly identifies capital of France
    """,model=evaluation_model,threshold=0.8,
          evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT])
    assert_test(testcase,[factual_metric])
    print(f'score is {factual_metric.score}')
    print(f'reasoning is {factual_metric.reason}')