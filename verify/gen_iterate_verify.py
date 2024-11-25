import json
import re
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
from tqdm import tqdm
import random
import argparse

# 添加项目路径
from utils.chat import *
from utils.retrieval import retrieval_class

# 参数解析
parser = argparse.ArgumentParser("Data Processing Script")
parser.add_argument("--in_path", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--language", type=str, required=True)
parser.add_argument("--is_llama", action="store_true")
args, unknown = parser.parse_known_args()

chat_class = ChatClass(args.model_path, args.model_path, is_llama=args.is_llama)
retrieval = retrieval_class()

question_pattern = r"Now the question is:\s*(.*?)(?=Please answer this question.)"
question_pattern2 = r"Now the question is:\s*(.*?)(?=Please provide a detailed answer to the question.)"
chinese_question_pattern = r"现在问题是：\s*(.*?)(?=请直接回答，不要有多余的内容。)"

english_prompt0 = "You are an expert who have read a lot of knowledge base. Please answer the question according to the content of the KB. "
english_prompt1 = "You can refer to some segments from the KB to help you answer the question. References: \n"
english_prompt2 = "Now the question is: "
english_prompt3_1 = "Please answer this question."
english_prompt3_2 = "Please provide a detailed answer to the question."

chinese_prompt0 = "你是一个法律知识专家，请你通过知识库直接回答问题"
chinese_prompt1 = "你可以参考知识库中的一些片段来帮助您回答问题。参考内容：\n"
chinese_prompt2 = "现在问题是："
chinese_prompt3 = "请直接回答，不要有多余的内容。"

def get_prompts(language):
    if language == "English":
        return english_prompt0, english_prompt1, english_prompt2, english_prompt3_1,english_prompt3_2
    elif language == "Chinese":
        return chinese_prompt0, chinese_prompt1, chinese_prompt2, chinese_prompt3, chinese_prompt3

def iter_jsonl(fname):
    with open(fname, "r") as fin:
        for line in fin:
            yield json.loads(line)

def process_data(data_lines,is_jsonl,language):
    new_data = []
    for data in tqdm(data_lines, desc="Processing JSONL Data"):
        try:
            if is_jsonl:
                data_0 = data['data'][0]
                data_1 = data['data'][1]
            else:
                data_0 = data['messages'][0]["content"]
                data_1 = data['messages'][1]["content"]
            question_match = re.search(question_pattern, data_0, re.DOTALL)
            question_match2 = re.search(question_pattern2, data_0, re.DOTALL)
            question_match_cn = re.search(chinese_question_pattern, data_0, re.DOTALL)
            kb_id=int(re.search(r'KB(\d+)', data_0).group(1))
            prompt0, prompt1, prompt2, prompt3_1, prompt3_2 = get_prompts(language)
            if language=="English":
                try:
                    question = question_match.group(1).strip()
                    prompt3=prompt3_1
                except:
                    question = question_match2.group(1).strip()
                    prompt3=prompt3_2
            else:
                question = question_match_cn.group(1).strip()
                prompt3=prompt3_1
            kb_id = int(re.search(r'KB(\d+)', data_0).group(1))
            query = [question]
            reference, _ = retrieval.get_references(query, kb_id, 8, 128, 'other', batch_size=256, bge_type="BAAI/bge-large-en-v1.5")
            user_content = f"User: {prompt0}<KB0>{prompt1}{reference}\n{prompt2}{question}\n{prompt3} \nAssistant: "
            res0 = chat_class.vllm_model_chat([user_content], temperature=0.2, top_p=0.8, max_tokens=3000)[0][0]
            prompt = f"""You are a teacher evaluating student responses. Remember:
        1. If the student's response fully aligns with the golden answer, start your response with 'The student response is correct because'.
        2. Otherwise, start your response with 'The student response is wrong because', and provide the ERROR TYPE!!! (e.g., does not answer the question directly, provide totally wrong information, provide only part of the information, provide unrelated information)
        3. Notice! You are NOT ALLOWED to directly point out the correct answer in your verification. You are NOT ALLOWED to directly point out the correct answer in your verification. You are NOT ALLOWED to directly point out the correct answer in your verification. You should only tell me the correctness and the error type.

        Now here are the materials:
        Reference: {reference}
        Question: {question}
        Golden Answer: {data_1}
        Student Response: {res0}
        Please generate your verification. You should start with the judgement, and then EXPLAIN the reason / the error type."""
            new_response = chat_class.vllm_model_chat([prompt], temperature=0.2, top_p=0.8, max_tokens=3000)[0][0]
            # if 'correct' not in new_response:
            if is_jsonl:
                data['data']=[prompt,new_response]
            else:
                data['messages'][0]["content"]=prompt
                data['messages'][1]["content"]=new_response
            new_data.append(data)
        except Exception as e:
            print(f"Error processing data: {e}")
            new_data.append(data)
    return new_data




def main():
    is_jsonl=True
    file_extension = Path(args.in_path).suffix
    if file_extension == ".jsonl":
        data_lines = list(iter_jsonl(args.in_path))
        is_jsonl=True
    elif file_extension == ".json":
        is_jsonl=False
        with open(args.in_path, "r") as infile:
            data_lines = json.load(infile)
    new_data = process_data(data_lines,is_jsonl,args.language)
    
    if not is_jsonl:
        part_number = int(args.in_path.split('part')[-1].split('.')[0])
        new_data = new_data * part_number
        random.shuffle(new_data)
        
    with open(args.out_path, 'w') as outfile:
        json.dump(new_data, outfile, indent=4, ensure_ascii=False)
    print("Data processing complete. Saved to", args.out_path)

if __name__ == "__main__":
    main()
