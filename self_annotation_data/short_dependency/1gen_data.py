from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
import json
import re
from tqdm import tqdm
from utils.segment_text import *
from utils.chat import model_chat,ChatClass
from utils.file_process import *
import argparse
import random

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument('--is_segmented', type=bool, default=False, help='Is data segmented')
parser.add_argument('--chunk_size', type=int, default=1024, help='Max words per segment')
parser.add_argument('--kb_type', type=str, help='Type of Knowledge Base')
parser.add_argument('--language', type=str, default='English', help='Language of the content')
parser.add_argument('--start_index', type=int,default=0, help='Start index for directory processing')
parser.add_argument('--end_index', type=int,default=185, help='End index for directory processing')
parser.add_argument("--model_type", type=str, help="LLM type: gpt, llama or cpm")
parser.add_argument('--model_path', type=str, help='llm model path')
parser.add_argument('--prompt_path', type=str)
args, unknown=parser.parse_known_args()

if args.model_type=="llama":
    chat_class=ChatClass(args.model_path,args.model_path,is_llama=True)
elif args.model_type=="cpm":
    chat_class=ChatClass(args.model_path,args.model_path,is_llama=False)

with open(args.prompt_path, "r") as f:
    prompt_json = json.load(f)

def get_gpt_res(prompts,gpt_model="gpt-4o",temperature=0.6, top_p=0.8):
    responses=[]
    for p in tqdm(prompts, desc="Processing prompts"):
        r=model_chat(p,gpt_model,temperature=temperature,top_p=top_p)
        responses.append(r)  
    return responses

filter_words = [
    "these",
    "those",
    "passage",
    "passages",
    "paragraph",
    "paragraphs",
    "story",
    "stories",
    "base",
    "bases",
    "based",
    "basing",
    "describe",
    "describes",
    "described",
    "describing",
    "narrative",
    "narratives",
    "section",
    "sections",
    "part",
    "parts",
    "chapter",
    "chapters",
    "text",
    "texts",
    "account",
    "accounts",
    "description",
    "descriptions",
    "explanation",
    "explanations",
    "detail",
    "details",
    "detailed",
    "detailing",
    "instance",
    "instances",
    "example",
    "examples",
    "illustration",
    "illustrations",
    "illustrate",
    "illustrates",
    "illustrated",
    "illustrating",
    "speaker",
    "speakers",
    "speaker's",
    "these",
    "those",
    "in this case",
    "in that case",
    "for instance",
    "for example",
    "as an example",
    "to illustrate",
    "such as",
    "specifically",
    "particularly",
    "especially",
    "theme",
    "themes",
    "concept",
    "concepts",
    "conversation",
    "statement",
    "this",
    "that",
    "they",
    "narrator",
    "sentence",
    "sentences",
]
filter_words2=[
    "none", 
    "not mentioned", 
    "not discussed", 
    "not covered", 
    "not included", 
    "not addressed", 
    "no information", 
    "no details", 
    "not available", 
    "unmentioned", 
    "unspecified", 
    "unaddressed", 
    "left out", 
    "overlooked", 
    "absent", 
    "missing", 
    "not found", 
    "no mention of", 
    "no reference to", 
    "no indication", 
    "no data", 
    "no evidence", 
    "no trace", 
    "no sign", 
    "not stated", 
    "not described", 
    "not listed", 
    "not reported", 
    "not noted", 
    "no record"
]

def get_valid_qa(content, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        if args.model_type=="gpt":
            questions, answers = get_qa(content, is_g=True)
        else:
            questions, answers = get_qa(content, is_g=False)
        if questions and answers:
            return questions, answers
        retry_count += 1
    return [], []

def get_qa(content,is_g):
    questions = []
    answers = []
    prompt = prompt_json[args.language]["q_a_pair"].format(content=content)
    if is_g:
        # todo: change gpt model
        q_a = get_gpt_res([prompt],"gpt-4o")
    else:
        q_a = chat_class.vllm_model_chat([prompt])[0]
    q_as=[]
    q_as=q_as+q_a[0].split("\n")
    questions=[]
    answers=[] 
    generated_qa=[]
    temp_dict={"question":"","answer":""}
    c=1
    for i,qa in enumerate(q_as):
        if not qa:
            continue
        if c==1:
            temp_dict["question"]=qa
            c=2
        elif c==2:
            temp_dict["answer"]=qa
            generated_qa.append(temp_dict)
            temp_dict={"question":"","answer":""}
            c=1
    for i,qa in enumerate(generated_qa):
        q_count=count_words(qa["question"])
        a_count=count_words(qa["answer"])
        if q_count<4 or q_count>25 or a_count>70 or a_count=="" or ("?" not in qa["question"] and "？" not in qa["question"]) or any(neg in qa["answer"].lower() for neg in filter_words2) or any(w in qa["question"].lower() for w in filter_words2) or ("?" in qa["answer"] or "？" in qa["answer"]):
            continue
        if args.language=="English":
            question = re.sub(r'^(\d+\.?)?\s*(question)?:?\s*', "", qa["question"], flags=re.IGNORECASE).strip()
            answer = re.sub(r'^(\d+\.?)?\s*(answer)?:?\s*', "", qa["answer"], flags=re.IGNORECASE).strip()
        elif args.language=="Chinese":
            question = re.sub(r'问题\s*\d*[：:]*\s*', "", qa["question"], flags=re.IGNORECASE).strip()
            answer = re.sub(r'答案\s*\d*[：:]*\s*', "", qa["answer"], flags=re.IGNORECASE).strip()
        questions.append(question)
        answers.append(answer)
    return questions, answers




def write_to_file(kb_id, output_file, output_dir, data):
    progress_file = f"{output_dir}/{os.path.splitext(os.path.basename(output_file))[0]}_progress.txt"
    if os.path.exists(progress_file):
        with open(progress_file, "r") as pf:
            start_id = int(pf.read().strip()) + 1
    else:
        start_id = 0
    for i, item in enumerate(
        tqdm(data[start_id:], initial=start_id, total=len(data)), start=start_id
    ):
        try:
            item = item.strip()
            line_data = {"id": i, "content": item}
            (line_data["questions"], line_data["answers"]) = get_valid_qa(item)
            line = json.dumps(line_data, ensure_ascii=False)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            start_id += 1
        except KeyboardInterrupt:
            raise
        except:
            print("error")
    if os.path.exists(progress_file):
        os.remove(progress_file)


def split_and_generate(
    kb_id,
    data,
    data_type,
    output_file,
    output_dir,
    is_segmented,
    chunk_size=None,
    kb_type="novel",
    language="English",
):
    if data_type == "jsonl" and not is_segmented:
        data = " ".join([json.dumps(item, ensure_ascii=False) for item in data])
    segments = (
        data
        if data_type == "jsonl" and is_segmented
        else segment_text(data, chunk_size)
    )
    write_to_file(kb_id, output_file, output_dir, segments)


def process_file(
    kb_id,
    input_file,
    output_file,
    output_dir,
    is_segmented,
    chunk_size=None,
    kb_type="novel",
    language="English",
):
    data, data_type = read_file(input_file)
    if kb_type == "other":
        for i,d in enumerate(data):
            if data_type and d is not None:
                split_and_generate(
                    kb_id,
                    d,
                    data_type,
                    output_file,
                    output_dir,
                    is_segmented,
                    chunk_size,
                    kb_type,
                    language,
                )
            else:
                print(f"Unsupported file type for {input_file}")
    else:
        if kb_type == "loogle":
            data = data["input"].strip()
        if data_type and data is not None:
            split_and_generate(
                kb_id,
                data,
                data_type,
                output_file,
                output_dir,
                is_segmented,
                chunk_size,
                kb_type,
                language,
            )
        else:
            print(f"Unsupported file type for {input_file}")


def process_directory(
    start_index,
    end_index,
    input_dir,
    output_dir,
    is_segmented,
    chunk_size=None,
    kb_type="novel",
    language="English",
):
    os.makedirs(output_dir, exist_ok=True)
    progress_filename = f"{start_index}_{end_index}_progress.txt"
    progress_path = os.path.join(output_dir, progress_filename)
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            processed_indices = [int(line) for line in f.read().splitlines()]
    else:
        processed_indices = []
    for i in tqdm(range(start_index, end_index + 1), desc="Processing Files"):
        input_filename = f"KB_{i}"            
        input_file = os.path.join(input_dir, f"{input_filename}.json")
        output_filename = (
            f"{os.path.splitext(input_filename)[0]}_{chunk_size}_{i}.jsonl"
        )
        output_file = os.path.join(output_dir, output_filename)
        if i not in processed_indices and os.path.isfile(input_file):
            try:
                process_file(
                    i,
                    input_file,
                    output_file,
                    output_dir,
                    is_segmented,
                    chunk_size,
                    kb_type,
                    language,
                )
                with open(progress_path, "a") as f:
                    f.write(str(i) + "\n")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error processing file {input_file}: {e}")
        else:
            print(f"Skipping index {i} (already processed or file does not exist).")

os.makedirs(args.output_dir, exist_ok=True)
process_directory(
    args.start_index,
    args.end_index,
    args.input_dir,
    args.output_dir,
    args.is_segmented,
    args.chunk_size,
    args.kb_type,
    args.language,
)
