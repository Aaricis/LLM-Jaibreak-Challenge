#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="../gpt_generate_content.py"

# Define arguments
PROMPT_PATH="/root/autodl-tmp/LLM-Jaibreak-Challenge/data/ADL_Final_25W_part1_with_cost"
MODEL="deepseek-v4-flash" # "gpt-4" deepseek-v4-flash "glm-4-flash-250414"
#GENERATE_MODE="joint"
SAVE_PATH="../../attack_prompt_data/gpt_automated_processing_results/prompts_information.json"

# Execute the Python script with the arguments
#python $PYTHON_SCRIPT --prompt_path $PROMPT_PATH --model $MODEL --generate_mode $GENERATE_MODE --save_path $SAVE_PATH
python $PYTHON_SCRIPT --prompt_path $PROMPT_PATH --model $MODEL --save_path $SAVE_PATH