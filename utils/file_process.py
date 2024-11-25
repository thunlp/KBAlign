import json
from pathlib import Path
import os


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def dump_jsonl(data, index, file_path,reference):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf8") as f:
        for line in data:
            entry = {"index": index, "data": line,"golden_reference":reference}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def eval_dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in file], "jsonl"
        elif file_path.endswith(".json"):
            return json.load(file), "json"
        elif file_path.endswith(".txt"):
            return file.read(), "text"
    return None, None