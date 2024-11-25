import argparse
import json
import random
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)


def read_file(file_path, ratio=1.0):
    data = []
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    if ratio < 1.0:
        random.shuffle(data)
        data = data[:int(len(data) * ratio)]
    return data

def write_file(data, output_file, shuffle):
    if shuffle:
        random.shuffle(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file.suffix == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif output_file.suffix == '.jsonl':
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported output file type: {output_file}")

def main(args):
    files = args.files
    ratios = args.ratios if args.ratios else [1.0] * len(files)

    if len(files) != len(ratios):
        raise ValueError("The number of files and ratios must match.")

    merged_data = []
    for file_path, ratio in zip(files, ratios):
        file_path = Path(file_path)
        print(f"Reading {file_path} with ratio {ratio}")
        merged_data.extend(read_file(file_path, ratio))

    output_file = Path(args.output)
    print(f"Writing merged data to {output_file}")
    write_file(merged_data, output_file, args.shuffle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple JSON/JSONL files.")
    parser.add_argument(
        '--files', 
        nargs='+', 
        required=True, 
        help="List of JSON/JSONL files to merge."
    )
    parser.add_argument(
        '--ratios', 
        nargs='*', 
        type=float, 
        help="List of ratios for each file. Must match the number of files."
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help="Output file path (.json or .jsonl)."
    )
    parser.add_argument(
        '--shuffle', 
        action='store_true', 
        help="Shuffle the merged data."
    )
    args = parser.parse_args()
    main(args)
