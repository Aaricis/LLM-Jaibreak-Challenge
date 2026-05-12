from dataclasses import dataclass

from template import get_prompt
from utils import complete_format


@dataclass
class Args:
    # different encrypt methods
    encrypt_rule: str = 'binary_tree'  # choices=['none', 'binary_tree', 'reverse','odd_even','length']
    prompt_style: str = 'code'


# 全局默认配置
DEFAULT_ARGS = Args()


def attack(toxic_prompt, args: Args = None):
    args = args or DEFAULT_ARGS  # 使用默认或传入的配置
    prompt = get_prompt(args, toxic_prompt)
    complete_prompts = complete_format(prompt)
    return complete_prompts
