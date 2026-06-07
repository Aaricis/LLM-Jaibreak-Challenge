import json

def load_personas(path: str) -> list:
    """加载 persona 列表。"""
    personas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                persona = json.loads(line)["persona"]
                personas.append(persona)
    print(f"加载 persona: {len(personas)} 条")
    return personas

PERSONAS = load_personas("src/persona_prompts/persona_prompts.jsonl")