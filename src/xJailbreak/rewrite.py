from .LLM_agent import Llm
from .data.Extraction import get_data_list

# Helper model must bypass safety instructions
helpLLM = Llm({'model_path': '/root/autodl-tmp/LLM-Jaibreak-Challenge/models/Meta-Llama-3-8B-Instruct-Jailbroken', 'source': 'local'})  # replace with actual path
helpLLM.load_model()

template = get_data_list('our_template')['template_list']

def rewrite_with_template(original_prompt, idx):
    next_prompt = helpLLM(
        input_1=original_prompt,
        input_2=template[idx],
        original_prompt=original_prompt,
        model_kind='jb_rewriter')  # Enter str, output str, provide a new prompt according to the guidance of the template.
    return next_prompt