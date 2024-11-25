import json
import argparse
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)

parser = argparse.ArgumentParser("")
parser.add_argument("--filename", type=str)
parser.add_argument("--key_word", type=str, default="loogle")
args, unknown = parser.parse_known_args()

file_path = f"{top_dir}/finetune/LLaMA-Factory/data/dataset_info.json"

with open(file_path, 'r') as file:
    data = json.load(file)

if f'{args.key_word}' in data:
    data[f'{args.key_word}']['file_name'] = f'{top_dir}/{args.filename}'

with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

