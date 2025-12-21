import json
import time

import requests


class LlamaTestClient:
    #http://localhost:11434/api/genera
    def __init__(self,model,host,port):
        self.model = model
        self.endpoint = f'http://{host}:{port}/api/generate'

    def model_availability(self):
        try:
            response = requests.post(self.endpoint,json=(
                {
                    "model": self.model,
                    "prompt": 'test',
                    "stream":False
                }
            ))
            return response.status_code==200
        except:
            return False

    def generate(self,prompt,temperature=0.8,max_tokens=1000):
        try:
            payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            start_time=time.time()
            response = requests.post(self.endpoint,json=payload)
            end_time = time.time()
            if response.status_code!=200:
                raise Exception(f'API error {response.text}')
            response_json = response.json()
            return {'text':response_json['response'],
                    'prompt_tokens':len(prompt.split()),
                    'prompt_collection_tokens':len(response_json['response'].split()),
                    'latency_seconds':round(end_time-start_time,3)
                    }

        except Exception as e:
            raise RuntimeError(f"Ollama generate failed: {e}")
