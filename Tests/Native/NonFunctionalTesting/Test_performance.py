import time


def test_basic_throughput(llm_client):
    """
    We limit max_tokens so that response lengths are consistent to check throughput
    :param llm_client:
    :return:None
    """
    prompt='Hi, How are you'
    result=[]
    start_time=time.time()
    for i in range(3):
        response=llm_client.generate(prompt,max_tokens=20)
        result.append(response)

    end_time=time.time()
    total_time=end_time-start_time
    total_words=sum([r["prompt_collection_tokens"] for r in result])
    words_per_second=total_words/total_time
    requests_per_minute=3/(total_time/60)
    print(f'throughput is : {words_per_second} words per second')
    print(f'request rate is : {requests_per_minute} requests per minute')
    print(f'average response size is {total_words/3} words')




