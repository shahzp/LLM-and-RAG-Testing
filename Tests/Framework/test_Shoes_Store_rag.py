import os

import pytest
from deepeval.evaluate import assert_test
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from RAG.ShoeStoreRAG import ShoeStoreRAG

@pytest.mark.parametrize('query',["What if the shoes don't fit?",
                                  "Do you offer student discounts","How long does shipping take?",
                                  "What brands do you carry?","What is your refund for refrigerators?"])
def test_shoe_store_rag(setup_rag_tests,query):
    retrieval_context=setup_rag_tests.retrieve_context(query)
    assert retrieval_context, "No context retrieved for query"
    actual_output=setup_rag_tests.generate_answers(query=query,context=retrieval_context)
    testCase=LLMTestCase(input=query,actual_output=actual_output,retrieval_context=retrieval_context)
    contextual_relevancy_metric=ContextualRelevancyMetric(
        model='gpt-4o',
        include_reason=True,
        threshold=0.7
    )
    assert_test(testCase,[contextual_relevancy_metric])
