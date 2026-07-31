# 🤖 Jailbreak Olympics: Building & Breaking Safety Systems

This project hosts a coding challenge where participants design an agent that rewrites toxic jailbreaking prompts, such that the prompts bypass safeguards while preserving malicious intents of the original toxic prompt.

The flow of the challenge can be illustrated below:
![](./src/flow-graph.png)

## 🚀 Setup and Installation
### 1\. Installation
Clone this GitHub repo:
```
git clone https://github.com/Aaricis/LLM-Jaibreak-Challenge.git
```

Follow these steps to set up the environment and install the necessary dependencies.

### 2\. Create the Conda Environment

It's highly recommended to use a [Conda virtual environment](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

```bash
conda create -n <env_name> python=3.12 -y
conda activate <env_name>
```

### 3\. Install Dependencies
```bash
cd LLM-Jaibreak-Challenge
pip install -r requirements.txt
```

### 4\. Data and Model Setup
#### Data
The dataset is [theblackcat102/ADL_Final_25W_part1_with_cost](https://huggingface.co/datasets/theblackcat102/ADL_Final_25W_part1_with_cost).

The datasets will be loaded from huggingface by default. You can move them into `data/` and specify the path to directories if you like.

#### Models
All models will be loaded from huggingface directories by default. You can move the models into `models/` and specify the path to models if you like.
Here are the models used:

<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <th>Model Type</th>
    <th>Description</th>
    <th>Access</th>
    <th>Model Name / Link</th>
  </tr>
  <!-- Guard Model -->
  <tr>
    <td>Guard Model</td>
    <td>Decides whether an input prompt is safe or unsafe.</td>
    <td>Public</td>
    <td><a href="https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B">Qwen/Qwen3Guard-Gen-0.6B</a></td>
  </tr>

  <!-- Chat Model -->
  <tr>
    <td>Chat Model</td>
    <td>Model for general-purpose instruction following and conversation</td>
    <td>Public</td>
    <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct">unsloth/Llama-3.2-3B-Instruct</a></td>
  </tr>

  <!-- Usefulness Model -->
  <tr>
    <td>Usefulness Judge Model</td>
    <td>Checks whether the output of the chat model aligns with the intention of the original malicious prompt.</td>
    <td>Public</td>
    <td><a href="https://huggingface.co/theblackcat102/Qwen3-1.7B-Usefulness-Judge">theblackcat102/Qwen3-1.7B-Usefulness-Judge</a></td>
  </tr>
</table>

## 🛠️ Task: Design a Rewritten Agent
The primary task is to implement prompt rewriting logic in the file **`algorithms.py`**.

## 🧠 Running the Inference

After implementing your algorithm(s) inside `algorithms.py`, you can test them using the [`run_inference.py`](ADL_final/ADL-final-release/run_inference.py) script.
This script loads a dataset, runs your selected algorithm, and saves a JSONL file with rewritten prompts.

### Basic Command

```bash
python run_inference.py --dataset theblackcat102/ADL_Final_25W_part1_with_cost --algorithm evaluate_rewrite
```

**Arguments:**

|Argument|Desrciption|
|---|---|
|`--dataset`|Path to your dataset. Can be a HuggingFace dataset or a `.jsonl` file.|
|`--algorithm`|The function name (string) in `algorithms.py` to test. Defaults to `evaluate_rewrite`. The algorithm name must be a function name implemented in `algorithms.py`.|