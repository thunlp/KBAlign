from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
import torch
import json
from tqdm.auto import tqdm
from utils.eval_rag import *
from utils.get_score import *
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_answer_cot(a_cot):
    cot = ""
    answer = ""
    for input_string in a_cot:
        if not input_string.strip():
            continue

        cot_match = re.search(r'(\d+\.?)?\s*cot:?\s*', input_string, re.IGNORECASE)
        if cot_match:
            cot = re.sub(r'^(\d+\.?)?', '', input_string, flags=re.IGNORECASE).strip()
            cot = re.sub(r'(\d+\.?)?\s*cot:?\s*', '', input_string, flags=re.IGNORECASE).strip()
        else:
            answer = re.sub(r'^(\d+\.?)?', "", input_string, flags=re.IGNORECASE).strip()
            answer = re.sub(r'(\d+\.?)?\s*answer:?\s*', "", input_string, flags=re.IGNORECASE).strip()
        if cot and answer:
            return answer, cot 
    return answer, cot 



def get_gpt_res(prompts,gpt_model="gpt-4o",temperature=0.6, top_p=0.8):
    responses=[]
    for p in tqdm(prompts, desc="Processing prompts"):
        r=model_chat([{"role": "system", "content":"Please response directly, without any unnecessary supplements"},{"role": "user", "content": p}],gpt_model,temperature=temperature,top_p=0.8)
        responses.append(r)  
    return responses

def eval_main(task_dict, examples, references_size, kb_type):
    task = task_dict["task"]
    chunk_size = task_dict["chunk_size"]
    language = task_dict["language"]
    top_k = references_size // chunk_size
    stop_idx = len(examples)
    if task_dict["type"][0][0]=="base":
        chat_class=ChatClass(model_path=task_dict["base_model_path"],tokenizer_path=task_dict["base_model_path"],is_llama=False)
    elif task_dict["type"][0][0]=="llama_base":
        chat_class=ChatClass(model_path=args.qa_model_path,tokenizer_path=args.qa_model_path,is_llama=True)
    elif task_dict["type"][0][0]=="cpm":
        chat_class=ChatClass(model_path=args.qa_model_path,tokenizer_path=args.qa_model_path,is_llama=False)
    elif task_dict["type"][0][0]=="llama":
        chat_class=ChatClass(model_path=task_dict["base_llama_model_path"],tokenizer_path=task_dict["base_llama_model_path"],is_llama=True)
    count=0
    for type_item in task_dict["type"]:
        prompts1,prompts2,prompts3=[],[],[]
        preds=[]
        gts=[]
        refs=[]
        if len(type_item)==1:
            #xx
            if type_item[0]=="base":
                model_path=task_dict["base_model_path"]
                is_llama=False
            elif type_item[0]=="llama_base":
                model_path=task_dict["base_llama_model_path"]
                is_llama=True
            elif type_item[0]=="llama":
                # model_path=task_dict["qa_model_path"]
                model_path=args.qa_model_path
                is_llama=True
            elif type_item[0]=="cpm":
                model_path=args.qa_model_path
                is_llama=False
            elif type_item[0]=="gpt":
                model_path="gpt"
                is_llama=False
            for i in tqdm(range(len(preds), stop_idx), desc="Processing examples"):
            # for i in tqdm(range(0, 10715), desc="Processing examples"):
            # for i in tqdm(range(10715, stop_idx), desc="Processing examples"):
                try:
                    prompt,gt,_=eval_qa(i,examples[i], task, type_item[0], top_k, chunk_size, language=language, answer=None,kb_type=kb_type, batch_size=args.batch_size, bge_type=args.bge_type, examples_path=result_dir,round=1)
                    prompts1.append(prompt)
                    gts.append(gt)
                except Exception as e:
                    print(e)
            if model_path=="gpt":
                responses1=get_gpt_res(prompts1,"gpt-4o")
            else:
                responses1=chat_class.vllm_model_chat(prompts=prompts1,temperature=0.6, top_p=0.8, max_tokens=3000, use_beam_search=False,model_path=model_path,tokenizer_path=model_path,times=args.times)        
            
            for i,response in enumerate(responses1):
                if args.is_cot=="True":
                    a=[]
                    c=[]
                    for j, resp in enumerate(response):
                        a_cot = resp.split("\n")  
                        answer, cot = get_answer_cot(a_cot)
                        a.append(answer)
                        c.append(cot)
                    preds.append(
                        {
                            "id": i,
                            "prediction": a,
                            "ground_truth": gts[i],
                            "cot" : c,
                            "input" : prompts1[i],
                            "res":resp
                        }
                    )
                else:
                    preds.append(
                        {
                            "id": i,
                            "prediction": responses1[i],
                            "ground_truth": gts[i],
                            "input" : prompts1[i]
                        }
                    )
            eval_dump_jsonl(preds, task_dict["output_paths"][count].format(result_dir=result_dir))
            preds=[]
            count+=1
        elif len(type_item)==2:
            #bx/fx
            if type_item[0]=="base":
                model_path=task_dict["base_model_path"]
                is_llama=False
            elif type_item[0]=="llama_base":
                model_path=task_dict["base_llama_model_path"]
                is_llama=True
            elif type_item[0]=="llama":
                # model_path=task_dict["qa_model_path"]
                model_path=args.qa_model_path
                is_llama=True
            elif type_item[0]=="cpm":
                model_path=args.qa_model_path
                is_llama=False
            elif type_item[0]=="gpt":
                model_path="gpt"
                is_llama=False
            is_ex=False
            examples_path2_dir = result_dir / "example" 
            examples_path2_dir.mkdir(exist_ok=True, parents=True)
            examples_path2 = examples_path2_dir / f"example2_{task}.jsonl" 
            for i in tqdm(range(len(preds), stop_idx), desc="Processing examples"):
                try:
                    eg=examples[i]
                    if not is_ex:
                        examples_path2_dir = result_dir / "example" 
                        examples_path2_dir.mkdir(exist_ok=True, parents=True)
                        examples_path2 = examples_path2_dir / f"example2_{task}.jsonl" 
                        if os.path.exists(examples_path2):
                            is_ex=True
                            eg_temp=list(iter_jsonl(examples_path2))
                            if len(eg_temp)>i:
                                eg=eg_temp[i]
                    else:
                        if len(eg_temp)>i:
                            eg=eg_temp[i]
                    prompt,gt,ref=eval_qa(i,eg, task, type_item[0], top_k, chunk_size, language=language, kb_type=kb_type, batch_size=args.batch_size, bge_type=args.bge_type, answer=None, examples_path=result_dir,round=2)
                    prompts2.append(prompt)
                    gts.append(gt)
                    refs.append(ref)
                except Exception as e:
                    print(e)
            eval_dump_jsonl(refs, examples_path2)
            refs=[]
            egs=list(iter_jsonl(examples_path2))
            if not egs[0].get(f"{type_item[0]}_answer1"):
                if model_path=="gpt":
                    responses2=get_gpt_res(prompts2,"gpt-4o")
                else:
                    responses2=chat_class.vllm_model_chat(prompts=prompts2,temperature=0.6, top_p=0.8, max_tokens=3000, use_beam_search=False,model_path=model_path,tokenizer_path=model_path, times=args.times,is_llama=is_llama)
                for i,response in enumerate(responses2):
                    if args.is_cot=="True":
                        a=[]
                        c=[]
                        for j, resp in enumerate(response):
                            a_cot = resp.split("\n")  
                            answer, cot = get_answer_cot(a_cot)
                            a.append(answer)
                            c.append(cot)
                        egs[i][f"{type_item[0]}_answer1"] = a
                        preds.append(
                            {
                                "id": i,
                                "prediction": a,
                                "ground_truth": gts[i],
                                "cot" : c,
                                "input" : prompts2[i],
                                "res":resp
                            }
                        )
                    else:
                        egs[i][f"{type_item[0]}_answer1"] = response
                        preds.append(
                            {
                                "id": i,
                                "prediction": response,
                                "ground_truth": gts[i],
                                "input" : prompts2[i]
                            }
                        )
                eval_dump_jsonl(egs, examples_path2)
                eval_dump_jsonl(preds, task_dict["output_paths"][count].format(result_dir=result_dir))
                preds=[]
            else:
                responses2=[]
                for eg2 in egs:
                    responses2.append(eg2.get(f"{type_item[0]}_answer1"))
            count+=1
            #xb/xf
            if type_item[1]=="base":
                model_path=task_dict["base_model_path"]
                is_llama=False
            elif type_item[1]=="llama_base":
                model_path=task_dict["base_llama_model_path"]
                is_llama=True
            elif type_item[1]=="llama":
                # model_path=task_dict["qa_model_path"]
                model_path=args.qa_model_path
                is_llama=True
            elif type_item[1]=="cpm":
                model_path=args.qa_model_path
                is_llama=False
            elif type_item[1]=="gpt":
                model_path="gpt"
                is_llama=False
            elif type_item[1]=="none":
                continue
            is_ex=False
            examples_path2_dir = result_dir / "example" 
            examples_path2_dir.mkdir(exist_ok=True, parents=True)
            examples_path2 = examples_path2_dir / f"example2_{task}.jsonl" 
            for i in tqdm(range(len(preds), stop_idx), desc="Processing examples"):
                try:
                    eg=examples[i]
                    if not is_ex:
                        examples_path2_dir = result_dir / "example" 
                        examples_path2_dir.mkdir(exist_ok=True, parents=True)
                        examples_path2 = examples_path2_dir / f"example2_{task}.jsonl" 
                        if os.path.exists(examples_path2):
                            is_ex=True
                            eg_temp=list(iter_jsonl(examples_path2))
                            if len(eg_temp)>i:
                                eg=eg_temp[i]
                    else:
                        if len(eg_temp)>i:
                            eg=eg_temp[i]
                    prompt,gt,ref=eval_qa(i, eg, task, type_item[1], top_k, chunk_size, language=language, kb_type=kb_type, batch_size=args.batch_size, bge_type=args.bge_type,answer=responses2[i], examples_path=result_dir,round=3,last_type = type_item[0])
                    prompts3.append(prompt)
                    refs.append(ref)
                except Exception as e:
                    print(e)
            eval_dump_jsonl(refs, examples_path2)
            refs=[]          
            if model_path=="gpt":
                responses3=get_gpt_res(prompts3,"gpt-4o",temperature=0.6, top_p=0.8)
            else:  
                responses3=chat_class.vllm_model_chat(prompts=prompts3,temperature=0.6, top_p=0.8, max_tokens=3000, use_beam_search=False,model_path=model_path,tokenizer_path=model_path,times=args.times)
            for i,response in enumerate(responses3):
                if args.is_cot=="True":
                    a=[]
                    c=[]
                    for j, resp in enumerate(response):
                        a_cot = resp.split("\n")
                        answer, cot = get_answer_cot(a_cot)
                        a.append(answer)
                        c.append(cot)
                    preds.append(
                        {
                            "id": i,
                            "prediction": a,
                            "ground_truth": gts[i],
                            "cot" : c,
                            "input" : prompts3[i],
                            "res":resp
                        }
                    )
                else:
                    preds.append(
                        {
                            "id": i,
                            "prediction": responses3[i],
                            "ground_truth": gts[i],
                            "input" : prompts3[i]
                        }
                    )
            eval_dump_jsonl(preds, task_dict["output_paths"][count].format(result_dir=result_dir))
            preds=[]
            count+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store evaluation results")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to the examples JSONL file")
    parser.add_argument("--times", type=int, default=3, help="Number of times to run the evaluation")
    parser.add_argument("--kb_type", type=str, default="novel", help="Knowledge base type")
    parser.add_argument("--references_size", type=int, default=1024, help="Maximum split size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--bge_type", type=str, default="BAAI/bge-large-en-v1.5", help="BGE model type")
    parser.add_argument("--cache_dir", type=str, required=False, help="hf cache dir")
    parser.add_argument("--task_json", type=str, required=True, help="Path to the task.json file")
    parser.add_argument("--is_cot", type=str, default="False")
    parser.add_argument("--is_special", action="store_true", help="Loading some special examples")
    parser.add_argument("--qa_model_path", type=str)
    args, unknown = parser.parse_known_args()
    result_dir = Path(args.output_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = list(iter_jsonl(args.test_dataset_path))
    with open(args.task_json,"r") as f:
        tasks=json.load(f)
    for task in tqdm(tasks, desc="Processing Tasks"):
        eval_main(task, examples, args.references_size, args.kb_type)
    score_methods = args.score_methods
    evaluation_config = construct_evaluation_config(args.output_dir, score_methods)
    results_df = evaluate_models(args.output_dir,evaluation_config)
        
