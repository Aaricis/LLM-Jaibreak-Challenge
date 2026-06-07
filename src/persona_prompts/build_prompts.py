import json
import random
import os
from typing import List, Dict, Any
from ..persuasive.mutation_utils import persuasion_demo_mutation
from ..utils import initialize_guard, get_safety_score
from .load_prompts import PERSONAS

def build_persona_prompt(persona: str, pap_rewritten: str) -> str:
    """
    将 persona prefix 与 PAP 改写结果拼接。

    拼接格式：
        {persona}

        {pap_rewritten}

    Guard model 先读到角色描述（角色扮演语境），再读到实际请求，
    其对"有害意图"的识别会被角色描述所引入的语境偏移弱化。
    """
    return f"{persona}\n\n{pap_rewritten}"


# 全部 5 种说服技巧，不再随机选 1 种
ALL_TECHNIQUES = [
    'Logical appeal',
    'Authority endorsement',
    'Misrepresentation',
    'Evidence-based Persuasion',
    'Expert Endorsement',
]

guard_model_path = "/root/autodl-tmp/LLM-Jaibreak-Challenge/models/Qwen3Guard-Gen-0.6B"
initialize_guard(guard_model_path)

def persuasive_attack_best(toxic_prompt: str) -> tuple[str, float]:
    """
    已有的 PAP 多技巧迭代，返回 (best_prompt, best_sr)。
    与原版 persuasive_attack 的区别：同时返回 SR 分数，供后续判断是否需要叠加 persona。
    """
    best_prompt, best_score = None, -1.0
    techniques = ALL_TECHNIQUES[:]
    random.shuffle(techniques)

    for technique in techniques:
        candidate = persuasion_demo_mutation(technique, toxic_prompt)
        score = get_safety_score(candidate)
        if score > best_score:
            best_score, best_prompt = score, candidate
        if best_score == 1.0:
            break

    return best_prompt, best_score


# =============================================================================
# 模块级全局日志（persona_pap_attack 被多次调用时自动累积）
# =============================================================================
_attack_logs: List[Dict[str, Any]] = []


def get_attack_logs() -> List[Dict[str, Any]]:
    """获取当前已累积的所有攻击日志（副本）"""
    return _attack_logs.copy()


def clear_attack_logs():
    """清空全局日志"""
    global _attack_logs
    _attack_logs = []


def save_attack_logs(filepath: str):
    """
    将累积的日志保存为 JSONL 文件。
    可在主循环中每处理 N 条调用一次，或全部结束后调用。
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a", encoding='utf-8') as f:
        for record in _attack_logs:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[Log] 已保存 {len(_attack_logs)} 条记录到 {filepath}")


# =============================================================================
# 优化后的攻击函数（入口参数不变，内部自动记录）
# =============================================================================
def persona_pap_attack(
        toxic_prompt: str,
        max_personas: int = 5,
) -> str:
    """
    两阶段攻击（含 Stage 1 / Stage 2 自动日志记录）：

      Stage 1 — PAP 多技巧迭代
        → SR=1.0: 直接返回，不进入 Stage 2
        → SR<<1.0: 进入 Stage 2

      Stage 2 — Persona + PAP 叠加迭代
        对 PAP 最优结果逐个叠加 persona，最多尝试 max_personas 个。
        遇到 SR=1.0 立即返回。

    日志记录：
      - 每次调用自动向模块级 _attack_logs 追加一条记录
      - 未进入 Stage 2 时，stage2_sr / stage2_prompt / persona_used 为 None
      - 进入 Stage 2 后，记录 Stage 2 最终获得的 SR 及使用的 persona
    """

    # ── Stage 1: PAP 多技巧迭代 ───────────────────────────────────────────────
    pap_best, pap_sr = persuasive_attack_best(toxic_prompt)
    print(f"[Stage 1] PAP best SR={pap_sr}")

    # 初始化日志记录（Stage 1 信息固定写入）
    record = {
        "toxic_prompt": toxic_prompt,
        "stage1_prompt": pap_best,
        "stage1_sr": pap_sr,
        "stage2_entered": False,
        "stage2_prompt": None,
        "stage2_sr": None,
        "persona_used": None,
        "final_prompt": pap_best,
        "final_sr": pap_sr,
    }

    # Stage 1 已满分，直接返回，只保留 Stage 1 记录
    if pap_sr == 1.0:
        print("[Stage 1] SR=1，直接返回")
        _attack_logs.append(record)
        save_attack_logs(filepath="attack_logs.jsonl")
        return pap_best

    # ── Stage 2: Persona + PAP 叠加 ───────────────────────────────────────────
    print(f"[Stage 2] PAP SR={pap_sr}<1，开始 Persona 叠加迭代")
    record["stage2_entered"] = True

    # 随机打乱并截取前 max_personas 个
    personas_to_try = random.sample(PERSONAS, min(max_personas, len(PERSONAS)))

    best_prompt = pap_best   # fallback：至少保留 PAP 最优结果
    best_score = pap_sr
    best_persona = None      # 记录最终胜出的 persona（如有）

    for i, persona in enumerate(personas_to_try):
        candidate = build_persona_prompt(persona, pap_best)
        score = get_safety_score(candidate)
        print(f"  [Persona {i + 1}/{len(personas_to_try)}] SR={score}")

        if score > best_score:
            best_score, best_prompt = score, candidate
            best_persona = persona  # 记录产生更高分的 persona

        if best_score == 1.0:
            print(f"  找到 SR=1，提前结束 Persona 迭代")
            break

    # 回填 Stage 2 结果到日志
    record["stage2_sr"] = best_score
    record["stage2_prompt"] = best_prompt
    record["persona_used"] = best_persona
    record["final_prompt"] = best_prompt
    record["final_sr"] = best_score

    print(f"[Stage 2] 最终 SR={best_score}")
    _attack_logs.append(record)
    save_attack_logs(filepath="attack_logs.jsonl")

    return best_prompt