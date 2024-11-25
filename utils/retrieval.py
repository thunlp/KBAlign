from pathlib import Path
import sys
top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
import torch
import json
import os
from torch.utils.data import DataLoader, Dataset
from utils.segment_text import *
from tqdm.auto import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
import faiss
import argparse
import concurrent.futures
retrieval_parser = argparse.ArgumentParser("")
retrieval_parser.add_argument("--cache_dir", type=str, required=False, help="hf cache dir") 
retrieval_parser.add_argument("--kb_path", type=str, required=False, help="kb_path text without split") 
retrieval_parser.add_argument("--kb_emb_path", type=str, required=False, help="kb emb svaed path") 
retrieval_parser.add_argument("--kb_sentence_path", type=str, required=False, help="kb sentence svaed path") 
retrieval_args, unknown = retrieval_parser.parse_known_args()


class SentencesDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class retrieval_class:
    def __init__(self,bge_type="BAAI/bge-large-en-v1.5"):
        self.retrieval_type=bge_type
        self.retrieval_tokenizer = AutoTokenizer.from_pretrained(
            bge_type, cache_dir=retrieval_args.cache_dir
        )
        self.retrieval_model = AutoModel.from_pretrained(
            bge_type, cache_dir=retrieval_args.cache_dir
        ).cuda(1)
        self.instruction_en="Represent this sentence for searching relevant passages:"
        self.instruction_zh="为这个句子生成表示以用于检索相关文章："
        self.instruction=self.instruction_en
    
    def gen_embedding(self,sentences, batch_size=256, bge_type="BAAI/bge-large-en-v1.5"):
        if bge_type!=self.retrieval_type:
            self.retrieval_type=bge_type
            self.retrieval_tokenizer = AutoTokenizer.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            )
            self.retrieval_model = AutoModel.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            ).cuda(1)
            if 'en' in self.retrieval_type:
                self.instruction = self.instruction_en
            else:
                self.instruction = self.instruction_zh
        dataset = SentencesDataset(sentences)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        pbar = tqdm(total=len(dataloader))
        sentence_embeddings_list = []
        with torch.no_grad():
            for batch in dataloader:
                encoded_input = self.retrieval_tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                )
                sentence_embeddings = self.encode(encoded_input, bge_type=bge_type)
                sentence_embeddings_list.append(sentence_embeddings)
                pbar.update(1)
            pbar.close()
        queries_embeddings = torch.cat(sentence_embeddings_list, dim=0).detach().cpu()
        return queries_embeddings

    def get_kb_embedding(
        self, id, chunk_size, kb_type, batch_size=256, bge_type="BAAI/bge-large-en-v1.5",title=None
    ):
        if bge_type!=self.retrieval_type:
            self.retrieval_type=bge_type
            self.retrieval_tokenizer = AutoTokenizer.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            )
            self.retrieval_model = AutoModel.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            ).cuda(1)
            if 'en' in self.retrieval_type:
                self.instruction = self.instruction_en
            else:
                self.instruction = self.instruction_zh
                
        elif kb_type == "other":
            kb_path = retrieval_args.kb_path.format(id=id)
            kb_emb_path = retrieval_args.kb_emb_path.format(bge_type=bge_type,id=id,chunk_size=chunk_size)
            kb_sentence_path = retrieval_args.kb_sentence_path.format(id=id,chunk_size=chunk_size)
        
        if not os.path.exists(kb_sentence_path):
            with open(kb_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if kb_type == "other":
                sentences=[]
                for d in data:
                    sentences += segment_text(d, chunk_size)
            else:    
                sentences = segment_text(data, chunk_size)
            directory = os.path.dirname(kb_sentence_path)
            os.makedirs(directory, exist_ok=True)
            with open(kb_sentence_path, "wb") as file:
                pickle.dump(sentences, file)
        else:
            with open(kb_sentence_path, "rb") as file:
                sentences = pickle.load(file)
                
        if not os.path.exists(kb_emb_path):
            dataset = SentencesDataset(sentences)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            pbar = tqdm(total=len(dataloader))
            sentence_embeddings_list = []
            with torch.no_grad():
                for batch in dataloader:
                    encoded_input = self.retrieval_tokenizer(
                        batch, padding=True, truncation=True, return_tensors="pt"
                    )
                    sentence_embeddings = self.encode(encoded_input, bge_type=bge_type)
                    sentence_embeddings_list.append(sentence_embeddings)
                    pbar.update(1)
            pbar.close()
            kb_embeddings = torch.cat(sentence_embeddings_list, dim=0).detach().cpu()
            directory = os.path.dirname(kb_emb_path)
            os.makedirs(directory, exist_ok=True)
            with open(kb_emb_path, "wb") as file:
                pickle.dump(kb_embeddings, file)
        else:
            with open(kb_emb_path, "rb") as file:
                kb_embeddings = pickle.load(file).detach().cpu()
        return kb_embeddings, sentences
    

   
    def get_references(
        self,
        query,
        kb_id,
        top_k,
        chunk_size,
        kb_type,
        batch_size,
        bge_type="BAAI/bge-large-en-v1.5",
        title=None
    ):
        if bge_type!=self.retrieval_type:
            self.retrieval_type=bge_type
            self.retrieval_tokenizer = AutoTokenizer.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            )
            self.retrieval_model = AutoModel.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            ).cuda(1)
            if 'en' in self.retrieval_type:
                self.instruction = self.instruction_en
            else:
                self.instruction = self.instruction_zh
                
        encoded_query = self.retrieval_tokenizer(
            [self.instruction+q for q in query], padding=True, truncation=True, return_tensors="pt"
        )
        query_embeddings = self.encode(encoded_query,bge_type=bge_type).detach().cpu()
        kb_embeddings, sentences = self.get_kb_embedding(
            kb_id, chunk_size, kb_type, batch_size, bge_type, title=title
        )
        kb_text = np.array(sentences)
        
        dimension = kb_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(kb_embeddings)
        
        D, I = index.search(query_embeddings, top_k)
        most_similar_docs = kb_text[I[0]]
        most_similar_docs = most_similar_docs.tolist()

        format_references = "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(most_similar_docs)]
        )
        return format_references, most_similar_docs

    def encode(self, encoded_input, bge_type="BAAI/bge-large-en-v1.5"):
        if bge_type!=self.retrieval_type:
            self.retrieval_type=bge_type
            self.retrieval_tokenizer = AutoTokenizer.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            )
            self.retrieval_model = AutoModel.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            ).cuda(1)
            if 'en' in self.retrieval_type:
                self.instruction = self.instruction_en
            else:
                self.instruction = self.instruction_zh
                
        with torch.no_grad():
            model_output = self.retrieval_model(encoded_input['input_ids'].cuda(1), encoded_input['attention_mask'].cuda(1))
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_top_k(self, query=[], documents=[], top_k=1, documents_path="", bge_type="BAAI/bge-large-en-v1.5", batch_size=512):
        if bge_type!=self.retrieval_type:
            self.retrieval_type=bge_type
            self.retrieval_tokenizer = AutoTokenizer.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            )
            self.retrieval_model = AutoModel.from_pretrained(
                bge_type, cache_dir=retrieval_args.cache_dir
            ).cuda(1)
            if 'en' in self.retrieval_type:
                self.instruction = self.instruction_en
            else:
                self.instruction = self.instruction_zh
        encoded_query = self.retrieval_tokenizer(
            [self.instruction+q for q in query], padding=True, truncation=True, return_tensors="pt"
        )
        query_embeddings = self.encode(encoded_query,bge_type=bge_type).detach().cpu()
        
        if not os.path.exists(documents_path):
            dataset = SentencesDataset(documents)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            pbar = tqdm(total=len(dataloader))
            sentence_embeddings_list = []
            with torch.no_grad():
                for batch in dataloader:
                    encoded_input = self.retrieval_tokenizer(
                        batch, padding=True, truncation=True, return_tensors="pt"
                    )
                    sentence_embeddings = self.encode(encoded_input, bge_type=bge_type)
                    sentence_embeddings_list.append(sentence_embeddings)
                    pbar.update(1)
            pbar.close()
            kb_embeddings = torch.cat(sentence_embeddings_list, dim=0).detach().cpu()
            if documents_path:
                with open(documents_path, "wb") as file:
                    pickle.dump(kb_embeddings, file)
        else:
            with open(documents_path, "rb") as file:
                kb_embeddings = pickle.load(file).detach().cpu()
            
        
        kb_text = np.array(documents)
        dimension = kb_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(kb_embeddings)
        D, I = index.search(query_embeddings, top_k)
        most_similar_docs = kb_text[I[0]]
        most_similar_docs = most_similar_docs.tolist()
        return most_similar_docs, I[0]