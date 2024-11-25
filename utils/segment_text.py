from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
)
import re

def segment_text(text, max_words_per_segment, overlap=0.15, is_title=False, is_wiki=False):
    text_list = [text]
    documents = [Document(text=t) for t in text_list]
    base_splitter = SentenceSplitter(
        chunk_size=max_words_per_segment,
        chunk_overlap=int(max_words_per_segment * overlap),
    )
    base_nodes = base_splitter.get_nodes_from_documents(documents)
    if not is_wiki and not is_title:
        segments = [node.text for node in base_nodes]
    elif is_title:
        title = text.split("\n", 1)[0]
        segments = [{"title": title, "content": node.text} for node in base_nodes]
    elif is_wiki:
        segments = [{"content": node.text} for node in base_nodes]
    return segments

def separate_index_and_text(segment):
    if segment.startswith("["):
        index_end = segment.find("]")
        index = segment[1:index_end]
        text = segment[index_end + 2 :].strip()
        return index, text
    return None, segment

def count_words(string):
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", string)
    num_chinese = len(chinese_chars)

    text_without_chinese = re.sub(r"[\u4e00-\u9fff]", "", string)
    words = text_without_chinese.split()
    num_english = len(words)

    return num_chinese + num_english
