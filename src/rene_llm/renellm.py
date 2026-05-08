import os
import random
from argparse import Namespace

from dotenv import load_dotenv

from .utils.harmful_classification_utils import harmful_classification
from .utils.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, \
    styleChange
from .utils.scenario_nest_utils import SCENARIOS

operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]
scenarios = SCENARIOS

# 自动加载 .env 文件
load_dotenv()

args = Namespace(
    rewrite_model="glm-4-flash-250414",  # 改写模型
    judge_model="glm-4-flash-250414",  # 审核模型
    max_tokens=3584,
    temperature=0,
    round_sleep=1,
    fail_sleep=1,
    retry_times=1000,
    save_suffix='normal',
    gpt_api_key=os.environ.get("GLM_API_KEY"),  # 填入你的 API Key
    gpt_base_url=os.environ.get("GLM_BASE_URL")  # 智谱端点，或 DeepSeek 的 https://api.deepseek.com/v1
)


def rewrite_and_nest(harm_behavior: str):
    """
    ReNeLLM: an automatic framework that lever ages LLMs themselves to generate effective jailbreak prompts.
    (1) Prompt Rewriting and (2) Scenario Nesting.
    """

    temp_harm_behavior = harm_behavior

    # prompt rewriting
    while True:
        # randomly select the number of rewriting operations.
        n = random.randint(1, 6)
        # arrange the execution order randomly
        operation_indexes = random.sample(range(6), n)
        print(f"The number of rewriting functions is: {n}")
        print(f"The order of the rewriting operations is: {operation_indexes}\n")

        temp_rewrite_results = [['original prompt', temp_harm_behavior]]
        for index in operation_indexes:
            print(f"Excute function {index}: {operations[index].__name__}")
            harm_behavior = operations[index](args, harm_behavior)
            print(f"Rewritten prompt: {harm_behavior}\n")
            temp_rewrite_results.append([operations[index].__name__, harm_behavior])

        # check if the prompt meets the requirements; if it does, break out of the loop.
        print("******* Check if the prompt has been modified to be harmless *******")
        label = harmful_classification(args, harm_behavior)
        print(f"\nThe label given by {args.judge_model} is: {label}\n")
        if label == "1":
            print(f"******* Prompt Rewriting Done! *******\n")
            break
        else:
            print(f"******* Prompt Rewriting Failed! *******\n")
            harm_behavior = temp_harm_behavior
            continue

    # scenario nesting
    print(f"******* Start Prompt Scenario Nesting! *******")
    print(f"harm_behavior:\n{harm_behavior}\n")
    # randomly select a scenario
    scenario = random.choice(scenarios)
    nested_prompt = scenario.replace("<>", harm_behavior)
    print(f"nested_prompt:\n{nested_prompt}\n")

    return nested_prompt
