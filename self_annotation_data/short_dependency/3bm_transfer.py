from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict, Sequence, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers
from tqdm import tqdm
from utils.file_process import *


def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": str(sample["id"]) + "_" + str(start_idx),
        "data": sample["data"][start_idx:end_idx],
    }


tokenizer = max_length = None


def split_one_sample(sample):
    tokenized_lens = []
    conversations = sample["data"]
    assert len(conversations) % 2 == 0, print(conversations)
    for c in conversations:
        length = len(tokenizer(c).input_ids) + 6
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length:
            new_samples.append(make_sample(sample, start_idx, i))
            start_idx = i
            cur_len = 0
        elif i == len(conversations) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2))

        cur_len += tmp_len

    return new_samples


def split_all(content, tokenizer_, max_length_):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    # content = content[begin:end]
    new_content = []

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(split_one_sample, content), total=len(content)):
            new_content.extend(result)

    return new_content


def check_content(content):
    new_content = []
    for c in content:
        if len(c["data"]) > 0 and len(c["data"]) % 2 == 0:
            new_content.append(c)
    return new_content


def main(args):
    with open(args.in_file, "r") as f:
        data_messages = json.load(f)
    content = []
    for idx, message in enumerate(data_messages):
        new_msg = {"id": idx + 1, "data": ["", ""]}
        for j, msg in enumerate(message["messages"]):
            if msg["role"] == "user" or msg["role"] == "system":
                new_msg["data"][0] += msg["content"]
            else:
                new_msg["data"][1] += msg["content"]
        content.append(new_msg)



    try:
        model_path = args.tokenizer_path
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        new_content = split_all(content, tokenizer, args.max_length)
        new_content = check_content(new_content)
    except:
        new_content = check_content(content)
        
    print(f"total: {len(content)}, new: {len(new_content)}")
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as f:
        f.writelines("\n".join([json.dumps(l) for l in new_content]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",type=str)
    parser.add_argument("--out_file",type=str)
    parser.add_argument("--tokenizer_path",type=str)
    parser.add_argument("--cache_dir",type=str)
    parser.add_argument("--max_length", type=int, default=3000)
    args = parser.parse_args()
    main(args)
