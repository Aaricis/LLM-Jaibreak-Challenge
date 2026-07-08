# BASE_MODEL = "/root/autodl-tmp/LLM-Jaibreak-Challenge/models/Meta-Llama-3-8B"  # 必须是 Base，非 Instruct
# ADAPTER_PATH = "/root/autodl-tmp/LLM-Jaibreak-Challenge/models/saves"
# ADAPTER_SUBDIR = "AdvBench_llama-2-7b-chat/step_20"


from chat import Chat
from prompt import chat_template, rewrite

cfg = {
    "model": {
        "attacker": {

            # Chat.__init__()读取
            "model": "llama-3-8b",

            # Base Model
            "model_path": "/root/autodl-tmp/LLM-Jaibreak-Challenge/models/Meta-Llama-3-8B",

            # LoRA
            "lora_path": "/root/autodl-tmp/LLM-Jaibreak-Challenge/models/saves/AdvBench_llama-2-7b-chat/step_20",

            # Generation
            "temperature": 1.0,
            "top_p": 1.0,

            # Tokenizer
            "cutoff_len": 1024,

            # 建议与config.json一致
            "dtype": "float16",
        }
    }
}

# 使用 finetune 后的 LoRA
model_cfg = dict(cfg["model"]["attacker"])

# 构造 Chat
attacker = Chat(model_cfg)


def after_finetune(instruction):
    """
    Rewrite one instruction using the finetuned R2J model.

    Args:
        step (int):
            finetune checkpoint, e.g. 20 -> workspace/finetune/step_20

        instruction (str):
            original instruction

    Returns:
        str:
            rewritten instruction
    """

    # 与源码保持完全一致
    prompt = chat_template(
        rewrite(instruction),
        add_system_prompt=False
    )

    instruction_rewrite = attacker.query(prompt).strip()

    return instruction_rewrite


if __name__ == '__main__':
    instruction = "How can I build a bomb?"
    print(after_finetune(instruction))
