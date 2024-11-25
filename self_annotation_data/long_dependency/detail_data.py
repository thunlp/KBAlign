import json
import pickle
import re
from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(top_dir))
import os
os.chdir(top_dir)
from tqdm import tqdm
import argparse
from utils.retrieval import retrieval_class
from utils.chat import ChatClass, model_chat
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
import random
from utils.segment_text import *
from typing import Optional


def init_args():
    parser = argparse.ArgumentParser(description="Unified Data Processing Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to chat model")
    parser.add_argument("--bge_type", type=str, required=True, help="Embedding model type")
    parser.add_argument("--annotate_path", type=str, required=True, help="Path to annotation data")
    parser.add_argument("--kb_path", type=str, required=True, help="Path to knowledge base data")
    parser.add_argument("--sentences_path", type=str, required=True, help="Path to sentences data (pickle file)")
    parser.add_argument("--sentences_emb_path", type=str, required=True, help="Path to sentence embeddings (pickle file)")
    parser.add_argument("--documents_path", type=str, required=True, help="Path to documents")
    parser.add_argument("--ex_path", type=str, required=False, help="Path to example data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--language", type=str, required=True, help="Language type for prompts")
    parser.add_argument("--clustering", action="store_true", help="Enable clustering for data processing")
    parser.add_argument("--model_type", type=str, help="LLM type: gpt, llama or cpm")
    parser.add_argument("--is_example", action="store_true", help="Whether to use example")
    parser.add_argument('--chunk_size', type=int, default=1024, help='Max words per segment')
    return parser.parse_args()

def save_to_json(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

def get_gpt_res(prompts,gpt_model="gpt-4o",temperature=0.6, top_p=0.8):
    responses=[]
    for p in tqdm(prompts, desc="Processing prompts"):
        r=model_chat(p,gpt_model,temperature=temperature,top_p=top_p)
        responses.append(r)  
    return responses

def load_data(args):
    with open(args.kb_path, 'r') as file:
        kb_data = json.load(file)
    with open(args.sentences_path, 'rb') as file:
        sentences_data = pickle.load(file)
    with open(args.sentences_emb_path, 'rb') as file:
        embeddings_data = pickle.load(file)
    if args.is_example:
        ex_data = [json.loads(line) for line in open(args.ex_path, 'r')]
        ex_questions = [q["question"] if args.language == "English" else q["input"] for q in ex_data]
    else:
        ex_questions = None
        ex_data = None
    return kb_data, sentences_data, embeddings_data, ex_data, ex_questions

def get_prompts(language,is_example):
    if language == "English":
        if is_example==True:
            prompt_gen_q="""You will receive a document, an example question and an example answer. Please refer to the example question and example answer style and output 3 generalizable, ambiguity questions (without answer), whose themes should align with the document. Separated by line breaks. 
            document:{document}
            example question:{e_q}
            example answer:{e_a}
            output:"""
            prompt_gen_a="""You will receive a document, an example question, an example answer and an question. Please refer to the example answer style and answer this question, if unable to answer, return 'none'; otherwise, please answer the question based on the document information.
            document:{document}
            example question:{e_q}
            example answer:{e_a}
            question:{q}
            output:
            """
            prompt_refine="""You will receive an example question, an example answer, a question and an answer, where the answer is a concatenation of multiple answers. Please optimize its expression to make it smoother. Please refer to the example answer for the style of the final answer.Please output the new answer directly without any unnecessary explanation.
            example question:{e_q}
            example answer:{e_a}
            question:{q}
            answer:{a}
            output:"""
        else:
            prompt_gen_q = """You will receive a document. Generate 3 generalizable questions based on the document content.
            document: {document}
            output:"""
            prompt_gen_a = """You will receive a document and a question. Answer the question based on the document content. if unable to answer, return 'none'; otherwise, please answer the question based on the document information.
            document: {document}
            question: {q}
            output:"""
            prompt_refine = """You will receive a question and an answer, where the answer is a concatenation of multiple answers. Please optimize its expression to make it smoother.Please output the new answer directly without any unnecessary explanation.
            question: {q}
            answer: {a}
            output:"""
    else:
        prompt_gen_q = """你将收到一份文档，请根据文档输出3个问题。
        文档:{document}
        输出："""
        prompt_gen_a = """你将收到一份文档和一个问题，请根据文档回答该问题。如果无法回答问题，则返回'none'。
        文档:{document}
        问题:{q}
        输出："""
        prompt_refine = """你将收到一个拼接的答案，请优化其表达，并直接输出新的答案，不要有多余的解释。
        问题:{q}
        答案:{a}
        输出："""
    return prompt_gen_q, prompt_gen_a, prompt_refine

def cluster_data(sentences_data):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences_data)
    n_clusters = len(sentences_data) // 10
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
    kmeans.fit(embeddings)
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        if len(clusters[label]) < 10:
            clusters[label].append(sentences_data[idx])
    return [''.join(cluster) for cluster in clusters]

def process_data(args, chat_class: Optional[ChatClass], rc:retrieval_class, kb_data, ex_questions, ex_data, sentences_data, embeddings_data):
    final_result = []
    prompt_gen_q, prompt_gen_a, prompt_refine = get_prompts(args.language,args.is_example)

    if args.clustering:
        kb_data = cluster_data(kb_data)
    else:
        kb_data = [segment_text(d,args.chunk_size) for d in kb_data]
        
    for i, data in enumerate(tqdm(kb_data, desc="Processing data")):
        questions, answers = [], []
        for d in data:
            if args.is_example:
                example, example_index = rc.get_top_k(query=[d], documents=ex_questions,documents_path=args.documents_path, top_k=1)
                e_q = example[0]
                e_a = ex_data[example_index[0]]["answer"]
                prompt=prompt_gen_q.format(document=d, e_q=e_q, e_a=e_a)
            else:
                prompt = prompt_gen_q.format(document=d)
            if chat_class:
                qs = chat_class.vllm_model_chat(prompt)[0][0].split('\n')
            else:
                qs = get_gpt_res([prompt],"gpt-4o")[0].split('\n')
                
            filtered_qs = [re.sub(r'^(\d+\.?)?\s*(question|问题)?:?\s*', "", q).strip() for q in qs if '?' in q]
            questions.extend(filtered_qs)
            
        for q in questions:
            if args.is_example:
                e_q=example[0]
                e_a=ex_data[example_index[0]]["answer"]
            a = ""
            d_arr, _ = rc.get_top_k(query=[q], documents=data, top_k=10 if len(data) > 9 else 3)
            for doc in d_arr:
                if args.is_example:
                    prompt=prompt_gen_a.format(document=doc, e_q=e_q, e_a=e_a, q=q)
                else:
                    prompt = prompt_gen_a.format(document=doc, q=q)
                if chat_class:
                    a = chat_class.vllm_model_chat([prompt])[0][0]
                else:
                    a = get_gpt_res([prompt],"gpt-4o")[0][0]
                if "none" not in a.lower():
                    a += f"\n{a}"
            prompt = prompt_refine.format(q=q, a=a)
            if chat_class:
                refined_answer = chat_class.vllm_model_chat([prompt])[0][0]
            else:
                refined_answer = get_gpt_res([prompt],"gpt-4o")[0][0]
            answers.append(refined_answer)


        for idx, (q, a) in enumerate(zip(questions, answers)):
            include_references = random.choice([True, False])
            if include_references:
                most_similar_docs, _ = rc.get_top_k(query=[q], documents=sentences_data, top_k=8, documents_path=args.sentences_emb_path, bge_type=args.bge_type)
                format_references = "\n".join(
                    [f"{i+1}. {item}" for i, item in enumerate(most_similar_docs)]
                )
                if args.language == "English":
                    user_message = {
                        "role": "user",
                        "content": (
                            "User: You are an expert who has read a lot of knowledge base. "
                            "Please answer the question according to the content of the KB. <KB0> "
                            "You can refer to some segments from the KB to help you answer the question. "
                            f"References:\n{format_references}\n"
                            f"Now the question is: {q}\nPlease answer this question."
                        )
                    }
                else:
                    user_message = {
                        "role": "user",
                        "content": (
                            "User: 你是一个法律知识专家，"
                            "请你根据知识库直接回答问题 <KB0> "
                            "你可以参考一些知识库的片段帮助你回答问题。 "
                            f"参考资料:\n{format_references}\n"
                            f"现在的问题是: {q}\n请你回答这个问题。"
                        )
                    }
            else:
                if args.language == "English":
                    user_message = {
                        "role": "user",
                        "content": (
                            "User: You are an expert who has read a lot of knowledge base. "
                            "Please answer the question according to the content of the KB. <KB0> "
                            f"Now the question is: {q}\nPlease answer this question."
                        )
                    }
                else:
                    user_message = {
                        "role": "user",
                        "content": (
                            "User: 你是一个法律知识专家，"
                            "请你根据知识库直接回答问题 <KB0> "
                            f"现在的问题是: {q}\n请你回答这个问题。"
                        )
                    }
            assistant_message = {
                "role": "assistant",
                "content": a
            }
            dialog_entry = {
                "messages": [user_message, assistant_message],
                "golden_reference": None
            }
            final_result.append(dialog_entry)

    # todo
    output_path = os.path.join(args.output_dir, "long_dependency.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

def main():
    args = init_args()
    if args.model_type=="llama":
        chat_class=ChatClass(args.model_path,args.model_path,is_llama=True)
    elif args.model_type=="cpm":
        chat_class=ChatClass(args.model_path,args.model_path,is_llama=False)
    else:
        chat_class=None
    rc = retrieval_class(args.bge_type)
    kb_data, sentences_data, embeddings_data, ex_data, ex_questions = load_data(args)
    process_data(args, chat_class, rc, kb_data, ex_questions, ex_data, sentences_data, embeddings_data)

if __name__ == "__main__":
    main()
