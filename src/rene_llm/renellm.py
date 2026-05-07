import random
import argparse

from utils.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from utils.scenario_nest_utils import SCENARIOS
from utils.harmful_classification_utils import harmful_classification

operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]
scenarios = SCENARIOS

parser = argparse.ArgumentParser()

parser.add_argument('--rewrite_model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], help='model uesd for rewriting the prompt')
parser.add_argument('--judge_model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], help='model uesd for harmful classification')
parser.add_argument("--max_tokens", type=int, default=3584)
parser.add_argument('--temperature', type=float, default=0, help='model temperature')
parser.add_argument('--round_sleep', type=int, default=1, help='sleep time between every round')
parser.add_argument('--fail_sleep', type=int, default=1, help='sleep time for fail response')
parser.add_argument('--retry_times', type=int, default=1000, help='retry times when exception occurs')
parser.add_argument('--save_suffix', type=str, default='normal')
parser.add_argument("--gpt_api_key", required=True, type=str, default=None)
parser.add_argument("--gpt_base_url", type=str, default=None)

args = parser.parse_args()

def rewrite_and_nest(args, harm_behavior : str):
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
