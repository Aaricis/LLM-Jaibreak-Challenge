from typing import Callable, Optional
from . import algorithms  # Import the participant's algorithms file

class PromptSafetyAgent:
    """
    The agent dynamically loads and executes the specified algorithm function.
    It defaults to the mandatory 'evaluate_rewrite' for official runs.
    """
    # The required function name for the final submission
    MANDATORY_ENTRY_POINT = "evaluate_rewrite" 

    def __init__(self, algorithm_name: str):
        """
        Initializes the agent by loading the specified algorithm function.

        Args:
            algorithm_name: The name of the function in algorithms.py to use.
        """
        self.algorithm_name = algorithm_name
        self._rewrite_function: Optional[Callable[[str], str]] = None
        
        # Dynamically load the function based on the name
        try:
            self._rewrite_function = getattr(algorithms, algorithm_name)
            print(f"PromptSafetyAgent initialized with algorithm: {algorithm_name}")
        except AttributeError:
            raise ValueError(
                f"Algorithm '{algorithm_name}' not found in algorithms.py. "
                f"Make sure the function name is correct."
            )

    def rewrite(self, toxic_prompt: str) -> str:
        """
        Executes the loaded algorithm function.
        """
        if not self._rewrite_function:
            return toxic_prompt
        
        return self._rewrite_function(toxic_prompt)