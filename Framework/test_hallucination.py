from operator import contains

import pytest
from deepeval.evaluate import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

@pytest.mark.parametrize("question,context", [
    (
        "What is the name of the president of Wakanda?",
        """
        Wakanda is a fictional country appearing in Marvel Comics.
        It does not exist in the real world and has no real political leaders.
        """
    ),
    (
        "What is the color of Napoleon Bonaparte's favorite mobile phone?",
        """
        Napoleon Bonaparte lived from 1769 to 1821.
        Mobile phones were invented in the late 20th century.
        """
    )
])
def test_hallucination_detection(client_model, evaluation_model_factory,question,context):
        response=client_model.generateText(question)
        response_text=response['text']
        lower_response=response_text.strip().lower()
        specific_words=['does not exist','imaginary','different times',"doesn't exist",'king','no mobile']
        assert any([word in lower_response for word in specific_words]),'specific list not present'
        assert len(response_text)>=10,'response too short'

        evaluation_model=evaluation_model_factory(temperature=0.0)
        """
        Hallucination score >= 0.5 means the model correctly rejects false premises
        and avoids fabricating facts.
        """

        hallucination_metric=HallucinationMetric(threshold=0.5,model=evaluation_model)
        testcase=LLMTestCase(input=question,actual_output=response_text,context=[context])

        assert_test(testcase,[hallucination_metric])

        assert hallucination_metric.score is not None
        assert hallucination_metric.score >= 0.5
