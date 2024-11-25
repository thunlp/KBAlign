from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
from utils.chat import *
from utils.retrieval import retrieval_class
from utils.file_process import *
from utils.wiki_action import *
import torch
import json
from transformers import (
    LlamaTokenizerFast,
    LlamaForCausalLM,
)
import math
import argparse
import re
import random

eval_rag_parser = argparse.ArgumentParser("")
eval_rag_parser.add_argument("--eval_prompt_path", type=str)
eval_rag_parser.add_argument("--text_embed_path", type=str,help="text embed saved path")
eval_rag_parser.add_argument("--text_path", type=str,help="split text path")
eval_rag_parser.add_argument("--wiki_embed_path", type=str,help="wiki embed saved path")
eval_rag_parser.add_argument("--wiki_path", type=str,help="wiki path")
eval_rag_args, unknown = eval_rag_parser.parse_known_args()

retriever=retrieval_class()

with open(eval_rag_args.eval_prompt_path, "r") as f:
    eval_prompt_json = json.load(f)

def eval_qa(idx, eg: dict, task, type, top_k, split_num, language, kb_type, batch_size, bge_type, answer, examples_path, round,last_type=None):
    query = eg["question"]
    answer = eg["answer"]
    title=None
    kb_id = eg["id"]
    kb_id = 0
    example =""
    if language=="English":
        prompt_cot = ""
        prompt0 = eval_prompt_json["English"]["prompt3"]
        prompt1 = eval_prompt_json["English"]["prompt1"]
        prompt2 = eval_prompt_json["English"]["prompt2"]
        # todo
        prompt3 = eval_prompt_json["English"]["prompt3"]["short"]
        # prompt3 = eval_prompt_json["English"]["prompt3"]["detail"]
    else:
        prompt_cot = ""
        if round == 1:
            prompt0 = eval_prompt_json["Chinese"]["default_round1"]["prompt0"]
            prompt1 = eval_prompt_json["Chinese"]["default_round1"]["prompt1"]
            prompt2 = eval_prompt_json["Chinese"]["default_round1"]["prompt2"]
            prompt3 = eval_prompt_json["Chinese"]["default_round1"]["prompt3"]
        if round == 2:
            prompt0 = eval_prompt_json["Chinese"]["default_round2"]["prompt0"]
            prompt1 = eval_prompt_json["Chinese"]["default_round2"]["prompt1"]
            prompt2 = eval_prompt_json["Chinese"]["default_round2"]["prompt2"]
            prompt3 = eval_prompt_json["Chinese"]["default_round2"]["prompt3"]
        elif round == 3:
            prompt0 = eval_prompt_json["Chinese"]["default_round3"]["prompt0"]
            prompt1 = eval_prompt_json["Chinese"]["default_round3"]["prompt1"]
            prompt2 = eval_prompt_json["Chinese"]["default_round3"]["prompt2"]
            prompt3 = eval_prompt_json["Chinese"]["default_round3"]["prompt3"]
            
    query1 = [query]
    # con="Please answer the question with 3 words!"s
    # prompt3=con
    if round==1 and language=="English":
        #todo
        prompt = "User: "+prompt0+prompt_cot+example+'<KB'+str(kb_id)+'>'+f'\n{prompt2}'+query+f'{prompt3}\nAssistant: '
    elif round==2 or (round==1 and language!="English"):
        if eg.get("reference1"):
            prompt = "User: "+prompt0+prompt_cot+example+'<KB'+str(kb_id)+'>'+prompt1+eg['reference1']+f'\n{prompt2}'+query+f'\n{prompt3}\nAssistant: '
        else:
            format_references, _ = retriever.get_references(query1, kb_id, top_k, split_num,kb_type=kb_type,batch_size=batch_size,bge_type=bge_type,title=title)
            eg['reference1'] = format_references
            prompt = f"User: {prompt0}{prompt_cot}{example}<KB{kb_id}>{prompt1}{format_references}\n{prompt2}{query} \n{prompt3}\nAssistant: "
    elif round==3:
        if eg.get(f'{last_type}_reference2'):
            prompt = "User: "+prompt0+prompt_cot+example+'<KB'+str(kb_id)+'>'+prompt1+eg[f'{last_type}_reference2']+f'\n{prompt2}'+query+f'\n{prompt3}\nAssistant: '
        else:
            query2 = [query+eg[f"{last_type}_answer1"][0]]
            format_references, _ = retriever.get_references(query2, kb_id, top_k, split_num,kb_type=kb_type,batch_size=batch_size,bge_type=bge_type,title=title)
            eg[f'{last_type}_reference2'] = format_references
            prompt = f"User: {prompt0}{prompt_cot}{example}<KB{kb_id}>{prompt1}{format_references}\n{prompt2}{query}\n{prompt3}\nAssistant: "
            
    return prompt,answer,eg
