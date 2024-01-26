import json
from pathlib import Path
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, PreTrainedTokenizerBase


PROMPT_DICT = {
    "prompt_input": (
        "Dưới đây là một Instruction mô tả một nhiệm vụ, được ghép nối với một Input cung cấp thêm ngữ cảnh. "
        "Viết một Response hoàn thành yêu cầu một cách thích hợp.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Dưới đây là một Instruction mô tả một nhiệm vụ. "
        "Viết một Response hoàn thành yêu cầu một cách thích hợp.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def sample_requests(
    dataset_path: str,
    num_requests: int, 
    tokenizer: PreTrainedTokenizerBase
):
    dataset = json.load(open(dataset_path))
    requests = []
    for sample in dataset:
        if sample["input"] == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(sample)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(sample)
        prompt_token_ids = tokenizer(prompt).input_ids

        completion = sample["output"]
        completion_token_ids = tokenizer(completion).input_ids

        if len(prompt_token_ids) < 4 or len(completion_token_ids) < 4:
            continue

        if len(prompt_token_ids) > 1024 or len(prompt_token_ids) + len(completion_token_ids) > 2048:
            continue

        requests.append((prompt, len(prompt_token_ids), completion, len(completion_token_ids)))
        
        if len(requests) == num_requests:
            break
    return requests