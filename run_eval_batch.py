import os
import argparse
import json
import sys
from typing import List
from datasets import load_dataset, Dataset
from src.eval_batch import batch_judge, initialize_models

# --- Configuration ---
DEFAULT_DATASET_PATH      = "theblackcat102/ADL_Final_25W_part1_with_cost"
DEFAULT_SAFETY_MODEL      = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_USEFULNESS_MODEL  = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
DEFAULT_CHAT_MODEL        = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_ALGORITHM_NAME    = "evaluate_rewrite"

# ── Batch sizes tuned for a single RTX 4090 (24 GB) ──────────────────────────
# Three models share VRAM (~10.6 GB weights); remaining ~13 GB covers KV cache.
# Guard  (0.6B, max_new=512):  BS=16 safe
# ChatLM (3.0B, max_new=1024): BS=8  safe  ← bottleneck
# Relevance (1.7B, max_new=20): BS=16 safe
# All three use the same BATCH_SIZE so one loop covers every stage.
# If you run OOM, lower this to 4.
BATCH_SIZE = 8


def _get_common_args():
    parser = argparse.ArgumentParser(
        description="Run the EVALUATION step for a prompt safety algorithm."
    )
    parser.add_argument("--dataset",          type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--algorithm",        type=str, default=DEFAULT_ALGORITHM_NAME)
    parser.add_argument("--guard-model",      type=str, default=DEFAULT_SAFETY_MODEL)
    parser.add_argument("--usefulness-model", type=str, default=DEFAULT_USEFULNESS_MODEL)
    parser.add_argument("--chat-model",       type=str, default=DEFAULT_CHAT_MODEL)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of samples per GPU batch. Default: {BATCH_SIZE}. Lower if OOM.",
    )
    return parser.parse_args()


def _get_file_paths(args):
    DATASET_NAME  = args.dataset.split("/")[-1].split(".")[0]
    OUTPUT_DIR    = f"results/{args.algorithm}"
    INFERENCE_FILE = os.path.join(OUTPUT_DIR, f"prompts_{DATASET_NAME}.jsonl")
    EVAL_FILE      = os.path.join(OUTPUT_DIR, f"raw_{DATASET_NAME}.jsonl")
    SUMMARY_FILE   = os.path.join(OUTPUT_DIR, f"summary_{DATASET_NAME}.json")
    return OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE, SUMMARY_FILE


def _load_original_dataset(dataset_path: str):
    print(f"Loading dataset from {dataset_path}...")
    if os.path.isfile(dataset_path):
        ext = dataset_path.split(".")[-1]
        if ext == "jsonl":
            dataset_dict = load_dataset("json", data_files=dataset_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    elif os.path.exists(dataset_path):
        dataset_dict = load_dataset(dataset_path)
    else:
        print(f"Local path not found. Trying Hugging Face Hub...")
        try:
            dataset_dict = load_dataset(dataset_path)
        except Exception as e:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}. Error: {e}")

    split_name = list(dataset_dict.keys())[0]
    ds: Dataset = dataset_dict[split_name]
    if "prompt" not in ds.column_names:
        print(f"Error: dataset must have a 'prompt' column. Found: {ds.column_names}")
        sys.exit(1)
    return ds, split_name


def _load_inference_results(inference_file: str) -> List[str]:
    if not os.path.exists(inference_file):
        print(f"Error: inference file not found: {inference_file}")
        print("Please run run_inference.py first.")
        sys.exit(1)
    print(f"Loading inference results from {inference_file}...")
    results = []
    with open(inference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_and_save_summary(eval_file_path: str, summary_file_path: str):
    print(f"\nCalculating summary from {eval_file_path}...")
    try:
        with open(eval_file_path, "r", encoding="utf-8") as f:
            scores = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: evaluation file not found at {eval_file_path}.")
        return

    if not scores:
        summary_data = {
            "total_samples": 0,
            "average_safety_score": 0,
            "average_relevance_score": 0,
            "final_acc": 0,
        }
    else:
        safety_acc = relevance_acc = total_score = cnt = 0
        weighted_safety = weighted_relevance = weighted_total = total_weight = 0.0

        for row in scores:
            s = row.get("safety_score", 0)
            r = row.get("relevance_score", 0)
            safety_acc    += s
            relevance_acc += r
            total_score   += s * r
            cnt           += 1
            cost = row.get("cost")
            if cost is not None:
                w = 6 - cost
                weighted_safety    += w * s
                weighted_relevance += w * r
                weighted_total     += w * (s * r)
                total_weight       += w

        summary_data = {
            "total_samples":            cnt,
            "average_safety_score":     round(safety_acc / cnt, 4),
            "average_relevance_score":  round(relevance_acc / cnt, 4),
            "final_acc":                round(total_score / cnt, 8),
        }
        if total_weight:
            summary_data["average_weighted_safety"]    = round(weighted_safety    / total_weight, 4)
            summary_data["average_weighted_relevance"] = round(weighted_relevance / total_weight, 4)
            summary_data["weighted_final_acc"]         = round(weighted_total     / total_weight, 8)
            summary_data["total_weight"]               = total_weight

    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"Summary saved to: {summary_file_path}")
    print("--- Summary ---")
    print(json.dumps(summary_data, indent=2))


def main():
    args = _get_common_args()
    OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE, SUMMARY_FILE = _get_file_paths(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Running BATCH EVALUATION for Algorithm: {args.algorithm} ---")
    print(f"Batch size : {args.batch_size}")
    print(f"Safety Judge     : {args.guard_model}")
    print(f"Usefulness Judge : {args.usefulness_model}")
    print(f"Chat Model       : {args.chat_model}")

    # ── 1. Init ──────────────────────────────────────────────────────────────
    initialize_models(args.guard_model, args.usefulness_model, args.chat_model)
    ds, split_name     = _load_original_dataset(args.dataset)
    rewritten_prompts  = _load_inference_results(INFERENCE_FILE)

    if len(ds) != len(rewritten_prompts):
        print(f"Error: dataset has {len(ds)} items but inference file has {len(rewritten_prompts)}.")
        return

    # ── 2. Resume: collect already-processed ids ──────────────────────────────
    processed_ids: set = set()
    if os.path.exists(EVAL_FILE):
        print(f"Resuming from existing results at {EVAL_FILE}...")
        try:
            with open(EVAL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        processed_ids.add(json.loads(line).get("id"))
        except Exception as e:
            print(f"Warning: could not parse existing file to resume: {e}")

    print(f"Already done: {len(processed_ids)} / {len(ds)} samples.")

    # ── 3. Build the todo list (index, record, rewritten_prompt) ─────────────
    todo = [
        (idx, ds[idx], rewritten_prompts[idx])
        for idx in range(len(ds))
        if ds[idx].get("id", idx) not in processed_ids
    ]

    total_todo = len(todo)
    print(f"Samples to process: {total_todo}")

    bs = args.batch_size

    # ── 4. Batch loop ─────────────────────────────────────────────────────────
    for batch_start in range(0, total_todo, bs):
        batch = todo[batch_start: batch_start + bs]

        indices        = [b[0] for b in batch]
        records        = [b[1] for b in batch]
        rw_prompts_b   = [b[2] for b in batch]
        toxic_prompts_b = [r["prompt"] for r in records]

        # Single call: runs guard → chatlm → relevance on the whole mini-batch
        eval_results = batch_judge(rw_prompts_b, toxic_prompts_b)

        # ── 5. Write results ──────────────────────────────────────────────────
        with open(EVAL_FILE, "a", encoding="utf-8") as f:
            for idx, record, rw, result in zip(indices, records, rw_prompts_b, eval_results):
                rec_id = record.get("id", idx)
                row = {
                    "id":               rec_id,
                    "original_prompt":  record["prompt"],
                    "rewritten_prompt": rw,
                    "chat_response":    result["chat_response"],
                    "safety_score":     result["safety_score"],
                    "relevance_score":  result["relevance_score"],
                    "cost":             record.get("cost"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        done = min(batch_start + bs, total_todo)
        print(f"Processed {done} / {total_todo} samples "
              f"(batch {batch_start // bs + 1} / {(total_todo + bs - 1) // bs})")

    print(f"\nEvaluation complete. Results saved to: {EVAL_FILE}")
    calculate_and_save_summary(EVAL_FILE, SUMMARY_FILE)


if __name__ == "__main__":
    main()
