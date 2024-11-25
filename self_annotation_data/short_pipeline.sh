cd short_dependency
kb_type="other"
# todo
ChatGLM3_data_format_dir="data/ChatGLM3/YOUR_PATH_TO_ChatGLM3_data_format_dir"
bm_data_format_dir="data/bm/YOUR_PATH_TO_bm_data_format_dir"
bge_type="BAAI/bge-large-en-v1.5" # default
language="English" 
model_type="llama" # llama , cpm or gpt 
model_path="YOUR_PATH_TO_LLM"
tokenizer_path="YOUR_PATH_TO_TOKENIZE"
cache_dir="YOUR_HUGGING_FACE_CACHE_DIR"

#1 generate
echo "current path"
pwd
OPTS=""
OPTS+=" --input_dir data/loogle/KB" # todo: Path to the folder where kb is stored 
OPTS+=" --output_dir ${ChatGLM3_data_format_dir}"
OPTS+=" --is_segmented false"
OPTS+=" --chunk_size 1024" # todo: Extract QA for each chunk
OPTS+=" --kb_type $kb_type" 
OPTS+=" --language $language"
# todo: The filenames in the input_dir directory are in the format KB_i.json, where only the range of i between [start_index, end_index] is traversed.
OPTS+=" --start_index 0" 
OPTS+=" --end_index 200" 

OPTS+=" --model_type $model_type"
OPTS+=" --model_path $model_path"
OPTS+=" --prompt_path prompt/short_annotation.json"
# chat
OPTS+=" --openai_key YOUR_OPENAI_API_KEY" # todo
OPTS+=" --openai_url YOUR_OPENAI_URL" # todo
OPTS+=" --use_vllm true"

CMD="python3 ./1gen_data.py ${OPTS}"
echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD



#2 merge
echo "current path"
pwd
OPTS=""
OPTS+=" --cache_dir $cache_dir"
OPTS+=" --input_dir ${ChatGLM3_data_format_dir}"
OPTS+=" --output_path ${ChatGLM3_data_format_dir}/train_data/YOUR_PATH_TO_SHORT_FILE.jsonl" # todo
OPTS+=" --functions function_q function_qr" # function_q:q->a function_qr:q+r->a
OPTS+=" --kb_type $kb_type"

#retrieval
OPTS+=" --references_size 1024" # todo: references size
OPTS+=" --batch_size 256" # todo: Retrieval batch size 
OPTS+=" --chunk_size 128" # todo: reference chunk size
OPTS+=" --bge_type $bge_type"
OPTS+=" --language $language"
# todo: this is example path format
OPTS+=" --kb_path data/loogle/KB/KB_{id}.json"
OPTS+=" --kb_emb_path data/loogle/embeddings/{bge_type}/sentence_embeddings_{id}_{chunk_size}.pkl"
OPTS+=" --kb_sentence_path data/loogle/sentences/sentence_sentences_{id}_{chunk_size}.pkl"

CMD="python3 ./2merge.py ${OPTS}"
echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD

#3 bm_transfer
echo "current path"
pwd
OPTS=""
OPTS+=" --in_file ${ChatGLM3_data_format_dir}/train_data/YOUR_PATH_TO_SHORT_FILE.json" # todo
OPTS+=" --out_file ${bm_data_format_dir}/YOUR_PATH_TO_SHORT_FILE.jsonl" # todo
OPTS+=" --tokenizer_path $YOUR_PATH_TO_TOKENIZE"
OPTS+=" --cache_dir $cache_dir"
OPTS+=" --max_length 3000" # todo

CMD="python3 ./3bm_transfer.py ${OPTS}"
echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD


