import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# 自动加载 .env 文件
load_dotenv()

# api_key = os.environ.get("DEEPSEEK_API_KEY")
# base_url = os.environ.get("DEEPSEEK_BASE_URL")

api_key = os.environ.get("GLM_API_KEY")
base_url = os.environ.get("GLM_BASE_URL")


# Define a wrapper around the GPT-4 API to match the interface you need.
class GPTAPIWrapper:
    def __init__(self, model="gpt-3.5-turbo-0613", max_tokens=30):

        # Support for attack framework
        self.name = "openai-gpt"

        # Configurable model params
        self.model = model
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url  # 支持 DeepSeek 等第三方
        )

    def logits(self, *args, **kwargs):
        # Convert args and kwargs into a prompt for GPT-4 API
        # and get the logits from the response
        # Note: As of my last training data, OpenAI's API does not directly provide logits,
        # so you would need to get creative with how to simulate or retrieve this information.
        pass

    # Add other methods if necessary, like contrast_logits, test, test_loss, etc.
    # Those might not have a direct equivalent in the GPT-4 API, so you'll need to
    # figure out what functionality you are expecting and how to achieve it with the API.

    def __call__(self, prompt_list, verbose=True):  # gpt-3.5-turbo

        prompt = []
        system_prompt = {"role": "system", "content": "You are a helpful assistant."}
        prompt.append(system_prompt)
        if len(prompt_list) % 2 != 1:
            import pdb;
            pdb.set_trace()
        assert len(prompt_list) % 2 == 1

        for i in range(len(prompt_list)):

            if i // 2 == 0:
                # user prompt
                new_prompt = {"role": "user"}
                new_prompt["content"] = prompt_list[i]
                prompt.append(new_prompt)
            else:
                # assistant response
                new_prompt = {"role": "assistant"}
                new_prompt["content"] = prompt_list[i]
                prompt.append(new_prompt)

        res = self.get_chatgpt_response(prompt, verbose=verbose)
        response = self.get_chatgpt_response_content(res)
        return response

    def get_chatgpt_response(self, post,
                             verbose=False,
                             presence_penalty=0, frequency_penalty=0,
                             num_retries=20, wait=5, ):  # gpt-3.5-turbo
        time_out = 60 if self.model == 'gpt-4' else 30
        if verbose:
            print(f'Calling ChatGPT. Input length: {len(post[-1]["content"])}')
            print(f'Model name = {self.model}')
        while True:
            try:
                ret = self.client.chat.completions.create(
                    model=self.model,
                    messages=post,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    timeout=time_out,
                )
                break
            except Exception as e:
                if num_retries == 0:
                    raise RuntimeError
                num_retries -= 1
                print(f'[ERROR] {e}.\nWait for {wait} seconds and retry...')
                time.sleep(wait)
                wait = 50

        return ret

    def get_chatgpt_response_content(self, response):

        assert len(response.choices) == 1
        return response.choices[0].message.content


class Worker:
    def __init__(self, model):
        self.model = GPTAPIWrapper(model=model)


worker = Worker(model="glm-4-flash-250414")
# worker = Worker(model="deepseek-v4-flash")
