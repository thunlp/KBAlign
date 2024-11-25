from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
from datasets import load_dataset
from collections import Counter
import string
import re
from tqdm import tqdm
import json
from pathlib import Path
import json
import evaluate
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from utils.chat import *
from typing import List
import argparse
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')

from rouge import Rouge
import random

get_score_parser = argparse.ArgumentParser("")
get_score_parser.add_argument("--cache_dir", type=str, required=False, help="hf cache dir")
get_score_args, unknown = get_score_parser.parse_known_args()


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score(pred: str, ground_truths) -> float:
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1 * 100


# def f1_score(prediction, ground_truth) -> tuple[float, float, float]:
def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# def bert_score(continuation: str, references: list[str], model) -> float:
def bert_score(continuation: str, references, model):
    sentences = [continuation] + references
    embeddings = model.encode(sentences)
    continuation_embedding = embeddings[0].reshape(1, -1)
    reference_embeddings = np.array(embeddings[1:])
    similarities = cosine_similarity(continuation_embedding, reference_embeddings)
    max_similarity = np.max(similarities)
    return max_similarity


def rougeL_score(continuation: str, reference: str, rouge) -> float:
    f = lambda text: normalize_answer(text).split()
    results = rouge.compute(
        predictions=[continuation],
        references=[reference],
        tokenizer=f,
        rouge_types=["rougeL"],
    )
    score = results["rougeL"]
    return score


def bleu_score(
    continuation: str,
    reference: List[str],
    bleu,
    with_penalty=False,
):
    f = lambda text: normalize_answer(text).split()
    results = bleu.compute(
        predictions=[continuation], references=[reference], tokenizer=f
    )
    bleu_avg = results["bleu"]

    precisions = results["precisions"]
    reference_length = results["reference_length"]

    for i in range(reference_length, 4):
        precisions[i] = precisions[reference_length - 1]
    bleu1 = precisions[0]
    bleu2 = precisions[1]
    bleu3 = precisions[2]
    bleu4 = precisions[3]

    brevity_penalty = results["brevity_penalty"]

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return (
            0.0 if brevity_penalty == 0 else bleu_avg / brevity_penalty,
            bleu1,
            bleu2,
            bleu3,
            bleu4,
        )


def extract_numbers(sentence):
    numbers = re.findall(r"\d+\.\d+|\d+", sentence)
    return numbers


def LLM_select_score(input, pred: str, ground_truths):
    retries = 3
    try:
        pattern = r'Now the question is: (.*?)Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone.'
        match = re.search(pattern, input, re.DOTALL)
        if not match:
            pattern = r'Now the question is: (.*?)Please answer this question.'
            match = re.search(pattern, input, re.DOTALL)
            if not match:
                return 50            
        question = match.group(1).strip()
        r = random.random()
        prompt = f"""Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' .
Question: {question}
groudtruth = {ground_truths[0]}
predict_answer = {pred}"""
        # res = model_chat(prompt,"meta_path",max_new_tokens=3000,temperature=0.0,top_p=1,repetition_penalty=1.1,finetuned_model=pipe, finetuned_tokenizer=None)
        res = model_chat(prompt,"gpt-4o",max_new_tokens=3000,temperature=0.0,top_p=1,repetition_penalty=1.1)
        if res.lower()=="true":
            return 100
        else:
            return 0
    except Exception as e:
        print(f"Error occurred: {e}")
        return 50

def select_score(pred, gt):
    try:
        pred = pred.lower()
        gt=gt
        pred_filtered = ''.join([char.lower() for char in pred if char.isalpha() and char.isascii()])
        gt_set = set([g.lower() for g in gt])
        pred_set = set(pred_filtered)
        if pred_set == gt_set:
            return 100
        else:
            return 0
    except Exception as e:
        print(e)
        return 0

    
def asqa_short_score(pred,gt):
    num=len(gt)
    right=0
    for i in gt:
        for j in i:
            if j.lower() in pred.lower():
                right+=1
                break
    return right/num*100
                
    

rouge = Rouge()
def get_score(pred, ground_truths, method="f1", evaluator=None, input="", base=None) -> float:
    def calculate_single_score(pred, ground_truths, method, evaluator):
        if pred == "":
            return 0.00
        if method == "f1":
            return qa_f1_score(pred, ground_truths)
        elif method == "bleu":
            return sentence_bleu([gt.split() for gt in ground_truths], pred.split(), weights=[0.25,0.25,0.25,0.25])*100
        elif method == "rouge":
            try:
                return rouge.get_scores(pred, ground_truths[0])[0]['rouge-l']['f']*100
            except:
                return 0
        elif method == "similarity":
            return bert_score(pred, ground_truths, evaluator) * 100
        elif method == "asqa_short":
            return asqa_short_score(pred, ground_truths)
        elif method == "Meteor":
            return meteor_score([word_tokenize(ground_truths[0])], word_tokenize(pred))*100
        elif method == "LLM_select":
            return LLM_select_score(input, pred, ground_truths)
        elif method == "select":
            return select_score(pred, ground_truths)
        else:
            raise NotImplementedError(f"Scoring method '{method}' is not implemented.")

    if isinstance(pred, str):
        return calculate_single_score(pred, ground_truths, method, evaluator)
    elif isinstance(pred, list):
        scores = [calculate_single_score(p, ground_truths, method, evaluator) for p in pred]
        return scores
        return sum(scores) / len(scores) if scores else 0.0
    else:
        raise ValueError("Prediction should be either a string or a list of strings.")

def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


text2vec = SentenceModel("shibing624/text2vec-base-multilingual")


def evaluate_models(output_dir,model_configs):
    results_df = pd.DataFrame()
    results_df2 = pd.DataFrame()
    results_df3 = pd.DataFrame()
    for config in model_configs:
        output_path = config["output_path"]
        score_methods = config["score_methods"]

        scores_dict = {method: [] for method in score_methods}
        #todo
        scores_dict2 = {method: [] for method in score_methods}
        scores_dict3 = {method: [] for method in score_methods}
        if Path(output_path).exists():
            preds1 = list(iter_jsonl(output_path))
            start_idx = len(preds1)
        else:
            start_idx = 0
            preds1 = []
        end_idx = len(preds1)
        
        #todo
        
        for i in tqdm(range(end_idx), desc=f"Processing with {os.path.basename(output_path)}"):
            if i < start_idx:
                idx=preds1[i]["id"]
                model_output = preds1[i]["prediction"]
                ground_truths = preds1[i]["ground_truth"]
                if isinstance(model_output, str):
                    model_output = [model_output]
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
            else:
                output_record = {
                    "prediction": model_output,
                    "ground_truth": ground_truths,
                }
                with open(output_path, "a") as file:
                    file.write(json.dumps(output_record) + "\n")
            for method in score_methods:
                if method == "f1":
                    score = get_score(model_output, ground_truths, method="f1")
                elif method == "bleu":
                    score = get_score(
                        model_output, ground_truths, method="bleu"
                    )
                elif method == "rouge":
                    score = get_score(
                        model_output, ground_truths, method="rouge"
                    )
                elif method == "similarity":
                    score = get_score(
                        model_output,
                        ground_truths,
                        method="similarity",
                        evaluator=text2vec,
                    )
                elif method == "asqa_short":
                    # todo
                    asqa_shorts=list(iter_jsonl(""))
                    score = get_score(model_output, asqa_shorts[i]["short_answers"], method="asqa_short")
                elif method == "Meteor":
                    score = get_score(model_output, ground_truths, method="Meteor")
                elif method == "LLM_select":
                    score = get_score(model_output, ground_truths, method="LLM_select", input=preds1[i]["input"])
                elif method == "select":
                    score = get_score(model_output, ground_truths, method="select", input=preds1[i]["input"])
                else:
                    raise ValueError(f"Unsupported scoring method: {method}")
                
                scores_dict[method].append(score)
                
                    
        model_results = pd.DataFrame({"Model": [output_path.split("/")[-1]]})
        for method, scores in scores_dict.items():
            model_results[method] = str(process_scores(scores))
            
        results_df = pd.concat([results_df, model_results], ignore_index=True)
    results_df.to_excel(
        f"{output_dir}/score.xlsx",
        index=False,
    )
    return results_df

def process_scores(scores):
    data = np.array(scores)
    column_means = np.mean(data, axis=0)
    overall_mean = np.mean(column_means)
    result = np.append(column_means, overall_mean)
    formatted_result = np.round(result, 2)
    return formatted_result

def filter_indices(output_path,num=1.5):
    preds1 = list(iter_jsonl(output_path))
    filtered_indices = [
        i for i, record in enumerate(preds1)
        if len(record["prediction"]) < num * len(record["ground_truth"])
    ]
    return filtered_indices

def construct_evaluation_config(directory, score_methods):
    evaluation_config = []
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            config = {
                "output_path": file_path,
                "score_methods": score_methods,
            }
            evaluation_config.append(config)
    return evaluation_config
sys.setrecursionlimit(10000)
if __name__ == "__main__":

    output_dirs = []
    score_methods = ["LLM_select"]
    for output_dir in output_dirs:
        evaluation_config = construct_evaluation_config(output_dir, score_methods)
        results_df = evaluate_models(output_dir,evaluation_config)
