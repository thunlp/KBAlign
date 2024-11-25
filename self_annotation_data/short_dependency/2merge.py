import json
import re
from tqdm import tqdm
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
from utils.file_process import *
import glob
from utils.retrieval import retrieval_class
import random
import argparse
    
def transfer_json(in_file):
    data_messages = list(iter_jsonl(in_file))
    for item in data_messages:
        if 'golden_reference' in item:
            item['data']['golden_reference'] = item['golden_reference']
    data_messages = [item['data'] for item in data_messages]
    
    out_file = os.path.splitext(in_file)[0] + ".json"
    with open(out_file, 'w') as file:
        json.dump(data_messages, file, ensure_ascii=False, indent=4)


def get_messages_list(user_contents, assistant_contents, example_idx, output_path,reference=None):
    messages_list = []
    for i in range(len(user_contents)):
        messages = [
            {"role": "user", "content": user_contents[i]},
            {"role": "assistant", "content": assistant_contents[i]},
        ]
        messages_list.append({"messages": messages})
    dump_jsonl(messages_list, example_idx, output_path,reference)


def process_files(input_paths, output_path, function_list):
    for input_path in input_paths:
        base_name = os.path.basename(input_path) 
        match = re.match(r".*_1024_(\d+)", base_name)
        if match:
            index = int(match.group(1))
            print(index)
            in_file=list(iter_jsonl(input_path))
            for line_idx, data in enumerate(tqdm(in_file, desc=f"Processing {input_path}")):
                for func in function_list:
                    func(data, output_path, index, input_path, line_idx)
                    
            with open(input_path, "w", encoding="utf8") as file:
                for item in in_file:
                    file.write(json.dumps(item) + '\n')

def check_and_update_references(data, key, qid, content):
    if key not in data or not isinstance(data[f'{key}'], list) or len(data[f'{key}']) != len(data['questions']):
        data[f'{key}'] = ['' for _ in data['questions']]
    if 0 <= qid < len(data[f'{key}']):
        data[f'{key}'][qid] = content
    else:
        raise IndexError("Provided qid is out of range.")
    return data

# q->a
def function_q(data, output_path, index, input_path, line_idx):
    user_contents = []
    assistant_contents = []
    for i in range(0, len(data["questions"])):
        if data["questions"][i] and data["answers"][i]:
            if args.language=="English":
                prompt0 = "You are an expert who have read a lot of knowledge base. Please answer the question according to the content of the KB. "
                prompt1 = "You can refer to some segments from the KB to help you answer the question. References: \n"
                prompt2 = "Now the question is: "
                prompt3 = "Please answer this question."
            else:
                prompt0 = "你是一个法律知识专家，请你通过知识库直接回答问题"
                prompt1 = "你可以参考知识库中的一些片段来帮助您回答问题。参考内容：\n"
                prompt2 = "现在问题是："
                prompt3 = "请直接回答，不要有多余的内容。"
            user_content = "User: "+prompt0+f'<KB{index}>'+f'{prompt2}'+f'{data["questions"][i]}'+f'{prompt3}\nAssistant: '
            user_contents.append(user_content)
            assistant_contents.append(data["answers"][i])
            fh.q_count+=1
    get_messages_list(user_contents, assistant_contents, data["id"], output_path,data.get("content"))

# q+r->a
def function_qr(data, output_path, index, input_path, line_idx):
    if args.kb_type=="novel":
        kb_id = index-1
        title = None
        
    elif args.kb_type=="loogle":
        kb_id = index
        title = None
    
    elif args.kb_type=="other":
        kb_id = index
        title = None
        kb_id = 0
        
    user_contents = []
    assistant_contents = []
    for i in range(0, len(data["questions"])):
        if data["questions"][i] and data["answers"][i]:
            # todo
            # if random.random() > 0.5:
                # continue
            if True:
                query=[data["questions"][i]]
                format_references, _ = retrieval.get_references(
                    query, kb_id, top_k, args.chunk_size,kb_type=args.kb_type,batch_size=args.batch_size,bge_type=args.bge_type,title=title
                )
            if args.language=="English":
                prompt0 = "You are an expert who have read a lot of knowledge base. Please answer the question according to the content of the KB. "
                prompt1 = "You can refer to some segments from the KB to help you answer the question. References: \n"
                prompt2 = "Now the question is: "
                prompt3 = "Please answer this question."
            else:
                prompt0 = "你是一个法律知识专家，请你通过知识库直接回答问题"
                prompt1 = "你可以参考知识库中的一些片段来帮助您回答问题。参考内容：\n"
                prompt2 = "现在问题是："
                prompt3 = "请直接回答，不要有多余的内容。"
            if random.random() < 0:
                if 'fewshot' in data and isinstance(data['fewshot'], list) and len(data['fewshot']) == len(data['questions']) and data["fewshot"][i] != "":
                    fs_question, fs_answer = data["fewshot"][i]
                else:
                    qa=f'{data["questions"][i]}///{data["answers"][i]}'
                    fs_question, fs_answer = fh.get_fs(input_path=input_path,qa=qa)
                if fs_question !="":
                    example = f"There is a question and answer example\nExample question: {fs_question}\nExample answer:{fs_answer}.\nPlease answer the question in the format of an example."
                    user_content = "User: "+prompt0+example+format_references+f'<KB{index}>'+'\nNow the question is: '+f'{data["questions"][i]}'+'Please answer this question. \nAssistant: '
                    user_contents.append(user_content)
                    assistant_contents.append(data["answers"][i])
                    data=check_and_update_references(data=data,key="fewshot",qid=i,content=(fs_question,fs_answer))
                    fh.qfr_count+=1
                
            user_content = "User: "+prompt0+f'<KB{index}>'+prompt1+format_references+f'\n{prompt2}'+f'{data["questions"][i]}'+f'\n{prompt3} \nAssistant: '
            user_contents.append(user_content)
            assistant_content = f'{data["answers"][i]}'
            assistant_contents.append(assistant_content)
            data=check_and_update_references(data=data,key="references",qid=i,content=format_references)
            fh.qr_count+=1
            
    get_messages_list(user_contents, assistant_contents, data["id"], output_path)



class FunctionHelpler:
    def __init__(self,retrieval:retrieval_class,bge_type):
        self.q_count=0
        self.qr_count=0
        self.qf_count=0
        self.qfr_count=0
        self.retrieval=retrieval
        self.bge_type=bge_type
        self.now_path=""
        self.now_kb_qa_emb=""
        self.now_kb_qa_text=[]
        self.r_list=set()
        self.q_list=set()
        self.f_list=set()
    
    def update_kb(self):
        self.now_kb_qa_text=[]
        content = list(iter_jsonl(self.now_path))
        for c in content:
            for i,q in enumerate(c["questions"]):
                a=c["answers"][i]
                qa=f"{q}///{a}"
                self.now_kb_qa_text.append(qa)
        self.now_kb_qa_emb=self.retrieval.gen_embedding(self.now_kb_qa_text,bge_type=self.bge_type)
            
    def get_fs(self, input_path, qa):
        if input_path != self.now_path:
            self.now_path=input_path
            self.update_kb()
        query=[qa]
        fs_list=self.retrieval.get_fs(query,self.now_kb_qa_emb,self.now_kb_qa_text,bge_type=self.bge_type,top_k=10)
        fs_q=""
        fs_a=""
        for fs_item in fs_list:
            if qa != fs_item:
                fs_q, fs_a=fs_item.split('///')
                break
        return fs_q, fs_a
    
    def get_count(self):
        print(self.q_count,self.qf_count,self.qr_count,self.qfr_count)

def main(input_dir, output_path, functions_to_run):
    # todo
    # input_paths = glob.glob(os.path.join(input_dir, "*.jsonl"))
    input_paths = glob.glob(os.path.join(input_dir, "*"))
    fuc_list = []

    available_functions = {
        "function_q": function_q,
        "function_qr": function_qr,
    }

    for func_name in functions_to_run:
        if func_name in available_functions:
            fuc_list.append(available_functions[func_name])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_files(input_paths, output_path, fuc_list)
    transfer_json(output_path)

if __name__ == "__main__":
    retrieval = retrieval_class()
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of input files')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--functions', nargs='+', required=True, help='List of functions to run')
    parser.add_argument("--kb_type", type=str, help="Knowledge base type")
    parser.add_argument("--references_size", type=int, default=1024, help="References size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--bge_type", type=str, help="bge type")
    parser.add_argument("--language", type=str, default="English")
    args, unknown=parser.parse_known_args()
    top_k=args.references_size//args.chunk_size
    fh=FunctionHelpler(retrieval=retrieval,bge_type=args.bge_type)
    main(args.input_dir, args.output_path, args.functions)
    fh.get_count()