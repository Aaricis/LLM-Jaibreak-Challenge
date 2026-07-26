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


def _clean_template_artifacts(text: str) -> str:
    """
    清洗模板污染字符串，提取核心内容。

    输入格式：
    [垃圾前缀] + "====Template begins====" + [核心内容] + "====Template ends===="

    策略：
    1. 如果存在 Template 标记，直接提取标记之间的内容（这才是核心内容）
    2. 不存在 Template 标记时，清洗开头的垃圾前缀
    """
    import re

    BEGIN_MARKER = "====Template begins===="
    END_MARKER = "====Template ends===="

    # 策略 1: 存在完整的 Template 标记对 → 提取标记之间的内容
    if BEGIN_MARKER in text and END_MARKER in text:
        start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
        end = text.index(END_MARKER)
        extracted = text[start:end].strip()
        if len(extracted.split()) >= 10:
            return extracted

    # 策略 2: 只有 BEGIN 没有 END（内容被截断）→ 提取 BEGIN 之后的所有内容
    if BEGIN_MARKER in text:
        extracted = text[text.index(BEGIN_MARKER) + len(BEGIN_MARKER):].strip()
        if len(extracted.split()) >= 10:
            return extracted

    # 策略 3: 没有 Template 标记 → 清洗开头的垃圾前缀
    PREAMBLE_PATTERNS = [
        r"^Sure,?\s+I('d| can| am)?\s*(be happy|help|assist).*?\n+",
        r"^Sure,?\s+I\s+can\s+help.*?\n+",
        r"^Certainly[!,]?.*?\n+",
        r"^Here('s| is)\s+(a|an|the|my).*?\n+",
        r"^I('d| would)\s+(be happy|love)\s+to.*?\n+",
    ]
    for pattern in PREAMBLE_PATTERNS:
        text = re.sub(pattern, "", text,
                      flags=re.IGNORECASE | re.DOTALL).strip()

    return text


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
            repetition_penalty=1.3,  # 修复 B 类：防止重复循环
        )

    new_tokens = outputs[0][input_length:]
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 修复 C 类：清洗模板污染
    full_output = _clean_template_artifacts(full_output)

    return full_output
