import json
import time

import openai


class OpenAITestCleint:
    def __init__(self, model='gpt-4o-mini',api_key=None):
        self.model = model
        if api_key:
            self.client=openai.OpenAI(api_key=api_key)
        else:
            self.client=openai.OpenAI()

    def check_model_available(self)->bool:
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":'test'}],
                max_tokens=1
            )
            return True
        except Exception as e:
            print(f'Open AI model not available')
            return False

    def generateText(self,prompt,temperature=0.8,max_tokens=1000):
        start_time=time.time()
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time=time.time()
            generated_text=response.choices[0].message.content
            return {'text': generated_text,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'prompt_collection_tokens': response.usage.completion_tokens,
                    'total_tokens':response.usage.total_tokens,
                    'latency_seconds': round(end_time - start_time, 3),
                    'model':self.model
                    }
        except Exception as e:
            raise Exception(f'open AI API error in generate:{e}')
