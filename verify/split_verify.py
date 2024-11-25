import json
import random
import argparse
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
parser = argparse.ArgumentParser("")
parser.add_argument("--in_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num", type=int, required=True)
args, unknown = parser.parse_known_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

file_extension = os.path.splitext(args.in_path)[-1]

def split_jsonl_file(input_path, num_parts):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    
    total_lines = len(lines)
    part_size = total_lines // num_parts
    
    chunks = [lines[i::num_parts] for i in range(num_parts)]
    
    for i in range(num_parts):
        output_path = f"{output_dir}/part{i+1}.jsonl"
        with open(output_path, 'w') as out_file:
            out_file.writelines(chunks[i])

def split_json_file(input_path, num_parts):
    with open(input_path, 'r') as f:
        array = json.load(f)

    chunks = [array[i::num_parts] for i in range(num_parts)]
    
    for i, chunk in enumerate(chunks):
        part_filename = f"{output_dir}/part{i + 1}.json"
        with open(part_filename, 'w') as f:
            json.dump(chunk, f, indent=4, ensure_ascii=False)

if file_extension == ".jsonl":
    split_jsonl_file(args.in_path, args.num)
elif file_extension == ".json":
    split_json_file(args.in_path, args.num)
else:
    raise ValueError("Unsupported file format. Please provide a .jsonl or .json file.")
