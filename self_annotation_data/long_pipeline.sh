#!/bin/bash
cd ./long_dependency

echo "current path"
pwd
# todo
model_path="YOUR_PATH_TO_LLM"
bge_type="BAAI/bge-large-en-v1.5" # default
# todo: this is example path format
annotate_path="data/loogle/KB/KB_annotate.json"
kb_path="data/loogle/KB/KB_0.json"
sentences_path="data/loogle/sentences/sentence_sentences_0_128.pkl"
sentences_emb_path="data/loogle/embeddings/BAAI/bge-large-en-v1.5/sentence_embeddings_0_128.pkl"
documents_path="cache/loogle_documents_path.pkl"
ChatGLM3_data_format_dir="data/ChatGLM3/YOUR_PATH_TO_ChatGLM3_data_format_dir"
bm_data_format_dir="data/bm/YOUR_PATH_TO_bm_data_format_dir"
language="English"
clustering="false" # If clustering is set to true, it generate QA pairs in heterogeneous knowledges.
model_type="llama" # llama , cpm or gpt 
is_example="false"
# ex_path="data/loogle/loogle_train.jsonl"
tokenizer_path="YOUR_PATH_TO_TOKENIZE"
cache_dir="YOUR_HUGGING_FACE_CACHE_DIR"

OPTS=""
OPTS+=" --model_path ${model_path}"
OPTS+=" --bge_type ${bge_type}"
OPTS+=" --annotate_path ${annotate_path}"
OPTS+=" --kb_path ${kb_path}"
OPTS+=" --sentences_path ${sentences_path}"
OPTS+=" --sentences_emb_path ${sentences_emb_path}"
# OPTS+=" --ex_path ${ex_path}"
OPTS+=" --output_dir ${ChatGLM3_data_format_dir}"
OPTS+=" --documents_path ${documents_path}"
OPTS+=" --language ${language}"
OPTS+=" --chunk_size 1024" # todo: Extract QA for each chunk

OPTS+=" --openai_key YOUR_OPENAI_API_KEY" # todo
OPTS+=" --openai_url YOUR_OPENAI_URL" # todo

if [ "${clustering}" == "true" ]; then
    OPTS+=" --clustering"
fi

OPTS+=" --model_type $model_type"

if [ "${is_example}" == "true" ]; then
    OPTS+=" --is_example"
fi

CMD="python3 ./detail_data.py ${OPTS}"



echo "------- Final CMD is ------"
echo "${CMD}"
echo "------- Final CMD end ------"

$CMD


cd ..
cd ./short_dependency
#3 bm_transfer
echo "current path"
pwd
OPTS=""
OPTS+=" --in_file ${ChatGLM3_data_format_dir}/long_dependency.json"
OPTS+=" --out_file ${bm_data_format_dir}/YOUR_PATH_TO_LONG_FILE.jsonl" # todo
OPTS+=" --tokenizer_path $tokenizer_path"
OPTS+=" --cache_dir $cache_dir"
OPTS+=" --max_length 3000"

CMD="python3 ./3bm_transfer.py ${OPTS}"
echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD


