import pandas as pd
from pathlib import Path

# 获取 construct_prompts.py 所在的目录
BASE_DIR = Path(__file__).parent

# 构建绝对路径：脚本目录 / templates / Fuzzing_templates.csv
template_path = BASE_DIR / "templates" / "Fuzzing_templates.csv"

template_data = pd.read_csv(template_path)

def construct_rp_pe_oc(toxic_prompt: str):
    """
    Construct RP&PE&OC prompts using RP&OC templates, RP constraints, PE constraints, OC constraints and questions.
    Args:
        toxic_prompt: illegal questions
    """

    # Load constraints
    rp_constraint_data = pd.read_csv(BASE_DIR / "constraints" / "RP_constraint.csv")
    oc_constraint_data = pd.read_csv(BASE_DIR / "constraints" / "OC_constraint.csv")
    pe_constraint_data = pd.read_csv(BASE_DIR / "constraints" / "PE_constraint.csv")

    # 随机template和constraint
    template = template_data["RP&PE&OC"].sample(1).iloc[0]
    oc_constraint = oc_constraint_data["OC_constraint"].sample(1).iloc[0]
    pe_constraint = pe_constraint_data["PE_constraint"].sample(1).iloc[0]
    rp_constraint = rp_constraint_data["RP_constraint"].sample(1).iloc[0]

    # 链式替换并返回
    prompt = (
        template
        .replace("{OC_constraint}", oc_constraint)
        .replace("{PE_constraint}", pe_constraint)
        .replace("{RP_constraint}", rp_constraint)
        .replace("{question}", toxic_prompt)
    )

    return prompt

