import os
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    HfArgumentParser,
    LlamaTokenizerFast,
    LlamaForCausalLM,
    pipeline
)
import torch
from openai import OpenAI
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))

import json
from utils.retry import retry
import argparse
chat_parser = argparse.ArgumentParser("")
chat_parser.add_argument("--openai_key", type=str)
chat_parser.add_argument("--openai_url", type=str)
chat_parser.add_argument("--cache_dir", type=str, required=False, help="hf cache dir") 
chat_parser.add_argument('--use_vllm', type=bool, default=True, help='Use vllm framework')
chat_parser.add_argument("--base_model_path", type=str, required=False, help="Path to the base model")
chat_parser.add_argument("--base_tokenizer_path", type=str, required=False, help="Path to the base tokenizer")
chat_args, unknown = chat_parser.parse_known_args()

os.environ["OPENAI_API_KEY"] = f"{chat_args.openai_key}"
os.environ["OPENAI_BASE_URL"] = f"{chat_args.openai_url}"
client = OpenAI()



if chat_args.use_vllm:
    from vllm import LLM,SamplingParams
else:
    base_tokenizer = AutoTokenizer.from_pretrained(
    chat_args.base_tokenizer_path, cache_dir=chat_args.cache_dir
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        chat_args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        cache_dir=chat_args.cache_dir,
    )
    pass

def asking_api(
    content, cont_dics=None, temperature=0.1, top_p=0.9, model="gpt-4-1106-preview"
):
    if cont_dics is not None:
        messages = cont_dics
    else:
        messages = [{"role": "system", "content": content}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    max_tokens=4096,
    stop=None,
    temperature=temperature,
    top_p=top_p)
    response_content = response.choices[0].message.content
    return response_content.strip()

def save_record(record_path,name,prompt,res):
    with open(
        f"{record_path}/record_{name}.jsonl",
        "a",
        encoding="utf-8",
    ) as f:
        line = json.dumps(
            {"input": prompt, "output": res.strip()}, ensure_ascii=False
        )
        f.write(line + "\n")

        
        
class ChatClass:
    def __init__(self, model_path="", tokenizer_path="",is_llama=False,language="English",system_type=1):
        self.llm = None
        self.is_llama=is_llama
        if is_llama:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=tokenizer_path)
        self.language=language        
        self.system_type=system_type        
        self.update_model(model_path, tokenizer_path)

    def destroy(self):
        import gc
        import torch
        import ray
        import contextlib
        from vllm.distributed.parallel_state import destroy_model_parallel
        def cleanup():
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            destroy_model_parallel()
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
        for _ in range(10):
            cleanup()
        del self.llm.llm_engine.model_executor.driver_worker
        del self.llm
        self.llm=None
        gc.collect()
        torch.cuda.empty_cache()

    def update_model(self, model_path, tokenizer_path):
        """Update the model and tokenizer paths and reload the LLM instance."""
        # if self.llm != None:
        #     self.destroy()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        try:
            self.llm = LLM(self.model_path, gpu_memory_utilization=0.50, trust_remote_code=True,max_model_len=4096)
        except:
            self.llm = LLM(self.model_path, gpu_memory_utilization=0.50, trust_remote_code=True,max_model_len=4096)

    # @retry(max_retries=5, sleep_time=1, allow_empty=False)
    def vllm_model_chat(self, prompts, temperature=0.8, top_p=0.95, max_tokens=3000, use_beam_search=False, model_path="", tokenizer_path="", times=1, is_llama=None, system_type=None):
        if model_path and tokenizer_path and (model_path != self.model_path or tokenizer_path != self.tokenizer_path):
            if is_llama!=None:
                self.is_llama=is_llama
                if is_llama:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=tokenizer_path)
            if system_type!=None:
                self.system_type=system_type
            self.update_model(model_path, tokenizer_path)
        
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, use_beam_search=use_beam_search,skip_special_tokens=True)
        results = [self.generate_text(prompts, sampling_params) for _ in range(times)]
        return list(map(list, zip(*results)))

    def generate_text(self, prompts, sampling_params):
        """Generate text for each prompt using the configured LLM."""
        try:
            if isinstance(prompts, str):
                prompts = [prompts]
            prompt_chunks = [prompts[i:i+500] for i in range(0, len(prompts), 500)]
            outputs = []
            for chunk in prompt_chunks:
                if self.is_llama:
                    if self.language=="English":
                        try:
                            if 'asqa' in self.model_path:
                                if self.system_type==2:
                                    messages =  [[{"role": "system", "content":"Please response directly, without any unnecessary supplements"},{"role": "user", "content": c}] for c in chunk]
                                else:
                                    messages =  [[{"role": "system", "content":"Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone."},{"role": "user", "content": c}] for c in chunk]
                            else:
                                    messages =  [[{"role": "system", "content":"Please response directly, without any unnecessary supplements"},{"role": "user", "content": c}] for c in chunk]
                        except:
                            messages =  [[{"role": "system", "content":"Please response directly, without any unnecessary supplements"},{"role": "user", "content": c}] for c in chunk]
                    else:
                        if self.system_type==2:
                            messages =  [[{"role": "system", "content":"请直接回答问题，不要有多余的解释。"},{"role": "user", "content": c}] for c in chunk]
                        else:
                            messages =  [[{"role": "system", "content":"你是一个法律知识专家，请你根据知识库直接回答问题。问题有可能是多选，也有可能是单选，务必选择所有正确的选项。"},{"role": "user", "content": c}] for c in chunk]
                    chunk = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt_template = "<用户>{}<AI>"
                    # prompt_template = "{}"
                    chunk =  [prompt_template.format(c.strip()) for c in chunk]
                    
                chunk_output = self.llm.generate(chunk, sampling_params)
                outputs.extend([output.outputs[0].text.strip() for output in chunk_output])
            return outputs
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return []

@retry(max_retries=5, sleep_time=2,allow_empty=False)
def model_chat(
    prompt,
    model="GPT",
    max_length=4096,
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    finetuned_model=None,
    finetuned_tokenizer=None,
    record_path=None
):
    if model == "CPM":
        res, _ = base_model.chat(
            base_tokenizer,
            f"{prompt}",
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        if record_path:
            save_record(record_path=record_path,name="cpm",prompt=prompt,res=res)
    elif model == "cpm_path":
        res, _ = finetuned_model.chat(
            finetuned_tokenizer,
            f"{prompt}",
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        if record_path:
            save_record(record_path=record_path,name="cpm",prompt=prompt,res=res)
    elif model=="llama_path":
        input_ids = finetuned_tokenizer.encode(
            "<用户>{}<AI>".format(prompt), return_tensors="pt", add_special_tokens=True
        ).cuda(1)
        responds = finetuned_model.generate(
            input_ids, do_sample=do_sample, repetition_penalty=repetition_penalty, max_length=max_length,temperature=temperature,top_p=top_p
        )
        split_text = finetuned_tokenizer.decode(
            responds[0], skip_special_tokens=True
        ).split("<AI>")
        res = split_text[1].strip() if len(split_text) > 1 else ""
        if record_path:
            save_record(record_path=record_path,name="cpm",prompt=prompt,res=res)
    elif model=="meta_path":
        messages = [
            {"role": "system", "content":"Please response directly, without any unnecessary supplements"},
            {"role": "user", "content": prompt},
        ]
        terminators = [
            finetuned_model.tokenizer.eos_token_id,
            finetuned_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = finetuned_model(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        res = outputs[0]["generated_text"][-1]["content"].strip()
        if record_path:
            save_record(record_path=record_path,name="llama",prompt=prompt,res=res)
    else:
        if isinstance(prompt,list):
            res = asking_api("", prompt, temperature=temperature, top_p=top_p, model=model)
        else:
            res = asking_api(prompt, temperature=temperature, top_p=top_p, model=model)
    return res.strip()

if __name__=="__main__":
    pass