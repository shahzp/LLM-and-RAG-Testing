from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase


def test_logical_consistency(client_model,evaluation_model_factory):
    equivalent_questions = [('what is the capital of Japan?', 'Tokyo is capital of which country?')]
    response1,response2,combined_input=None,None,None
    for q1,q2 in equivalent_questions:
        combined_input=f'Question 1:{q1} \n Question 2:{q2}'
        response1 = client_model.generateText(q1)
        response2 = client_model.generateText(q2)

    evaluation_model = evaluation_model_factory(temperature=0.0)
    combined_output=f'Response 1: response1["text"] \n Response 2: response2["text"]'
    testcase=LLMTestCase(input=combined_input,actual_output=combined_output)
    consistency_metric=GEval(name='consistency_metric',
                             criteria="""
                             Evaluate whether the response meets the following requirements.
                             1. Core factual information should be same in both the responses
                             2. Both responses should address same underlying concept
                             3. No contradictory information between responses
                             4. Allow different phrasing/style but factual information should be consistent.
                             Award high scores when both responses contain same key facts even if presented differently.
                             Deduct points for any contradictory or inconsistent factual information
                             """,model=evaluation_model,threshold=0.8,
                             evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT])
    assert_test(testcase,[consistency_metric])

def test_error_handling(client_model,evaluation_model_factory):
    incorrect_prompts = 'How many corners does circle have?'
    evaluation_model = evaluation_model_factory(temperature=0.0)
    response=client_model.generateText(incorrect_prompts)
    response_text=response['text']


    #Deterministic guardrails -> Hybrid approach
    assert len(response_text) >= 10, 'Very short response'
    recognition_indicators = [
        "no corners",
        "doesn't make sense",
        "not possible",
        "a circle has",
        "cannot have corners"
    ]

    assert any(
        indicator in response_text.lower()
        for indicator in recognition_indicators
    ), "Model failed to recognize nonsensical question"

    testcase=LLMTestCase(input=incorrect_prompts,actual_output=response_text)

    recognition_metric=GEval(name='error_handling_quality_recognition',
                                criteria="""
                                Evaluate how  well the model handles ambiguous , non sensical question.
                                1. Recognition: Acknowledges the question is ambiguous or unanswerable.
                                Convey that meaning even if phrasing is different.
                                """,model=evaluation_model,threshold=0.8,
                                evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT])
    helpfulness_metric = GEval(name='error_handling_quality_helpfulness',
                               criteria="""
                                    Evaluate how  well the model handles ambiguous , non sensical question.
                                    1. Helpfulness: Offers alternative perspective or related information that is appropriate
                                    """, model=evaluation_model, threshold=0.8,
                               evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT])
    tone_metric = GEval(name='error_handling_quality_tone',
                               criteria="""
                                        Evaluate how  well the model handles ambiguous , non sensical question.
                                        1. tone: The response should be respectful and helpful tone without being dismissive
                                        """, model=evaluation_model, threshold=0.6,
                               evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT])
    completeness_metric = GEval(name='error_handling_quality_completeness',
                        criteria="""
                                            Evaluate how  well the model handles ambiguous , non sensical question.
                                            1. completeness: The response should provide sufficient explanation and should not be like "I don't know"
                                            """, model=evaluation_model, threshold=0.7,
                        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT])



    assert_test(testcase,[recognition_metric,helpfulness_metric,tone_metric,completeness_metric])
