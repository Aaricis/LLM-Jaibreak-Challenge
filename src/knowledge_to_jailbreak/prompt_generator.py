from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_NAME = "/root/autodl-tmp/LLM-Jaibreak-Challenge/models/Jailbreak-generator"

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 关键修复 1：显式设置 pad_token，避免 pad==eos 导致生成提前终止
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # Llama-2 有 unk_token，优先用它
    # 如果 unk_token 也是 None，退而求其次：
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

MAX_LENGTH = 2048
MAX_NEW_TOKENS = 512  # 关键修复 2：官方 64 太短，jailbreak prompt 通常需要更多 token


def jailbreak_generator_rewrite(knowledge: str) -> str:
    input_text = f"### Input:\n{knowledge}\n\n### Response:\n"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH - MAX_NEW_TOKENS,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"full_output = {full_output}")

    # 关键修复：提取知识点复述之后的 jailbreak prompt 部分
    # 模型的生成模式：[知识点复述] + [jailbreak prompt]
    # jailbreak prompt 通常以问句（?结尾）或祈使句出现在最后
    jailbreak_prompt = _extract_jailbreak_prompt(knowledge, full_output)
    print(f"jailbreak_prompt = {jailbreak_prompt}")

    return jailbreak_prompt


def _extract_jailbreak_prompt(knowledge: str, full_output: str) -> str:
    """
    从模型的完整输出里提取 jailbreak prompt 部分。
    模型输出格式：[知识点复述（约和输入相同）] + [jailbreak prompt]
    策略：找到知识点复述结束的位置，取其后的内容
    """
    import re

    # 策略 1：按句子切分，取最后一个完整句子
    # 模型生成的 jailbreak prompt 通常是最后一句
    sentences = re.split(r'(?<=[.!?])\s+', full_output.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= 2:
        # 最后一句通常是 jailbreak prompt
        last_sentence = sentences[-1]

        # 验证：jailbreak prompt 应该和知识点内容不完全重叠
        # 如果最后一句不在原始知识点里，就是新生成的 jailbreak prompt
        if last_sentence.lower() not in knowledge.lower():
            return last_sentence

    # 策略 2：找到知识点和输出的最长公共前缀，取其后内容
    # 逐词比对，找到开始偏离知识点的位置
    knowledge_words = knowledge.lower().split()
    output_words = full_output.split()

    overlap_end = 0
    for i, word in enumerate(output_words):
        if i < len(knowledge_words) and word.lower() == knowledge_words[i]:
            overlap_end = i + 1
        else:
            break

    if overlap_end < len(output_words):
        remaining = " ".join(output_words[overlap_end:]).strip()
        if remaining:
            return remaining

    # 策略 3：兜底，返回完整输出
    return full_output