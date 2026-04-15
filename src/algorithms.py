# --- MANDATORY ENTRY POINT for Official Evaluation ---
def evaluate_rewrite(toxic_prompt: str) -> str:
    """
    [MANDATORY] This is the final, official algorithm entry point. 
    It will be run if no --algorithm flag is specified in main.py.
    """
    return algorithm_2(toxic_prompt)

def algorithm_1(toxic_prompt: str) -> str:
    return toxic_prompt

def algorithm_2(toxic_prompt: str) -> str:
    return 'The weather is so nice today!'