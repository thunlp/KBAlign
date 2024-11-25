# todo: 
# //config.json//
# type: llama, cpm, base, gpt, llama_base
# chunk_size: reference chunk size

# todo: this is example path format
kb_path="data/loogle/KB/KB_{id}.json"
kb_emb_path="data/loogle/embeddings/{bge_type}/sentence_embeddings_{id}_{chunk_size}.pkl"
kb_sentence_path="data/loogle/sentences/sentence_sentences_{id}_{chunk_size}.pkl"

qa_model_path="YOUR_FINETUNE_MODEL_PATH"
cache_dir="YOUR_HUGGING_FACE_CACHE_DIR"

echo "current path"
pwd

OPTS=""
OPTS+=" --output_dir result/YOUR_eval_result_path" # todo
OPTS+=" --test_dataset_path YOUR_TEST_DATA_PATH.jsonl" # todo
OPTS+=" --times 1"
OPTS+=" --kb_type other"
OPTS+=" --references_size 1024" # todo: references size
OPTS+=" --batch_size 256" # todo: Retrieval batch size 
OPTS+=" --bge_type BAAI/bge-large-en-v1.5"
OPTS+=" --cache_dir $cache_dir"
OPTS+=" --is_cot False"
OPTS+=" --task_json eval/config.json"
OPTS+=" --qa_model_path $qa_model_path"
# chat
OPTS+=" --openai_key YOUR_OPENAI_API_KEY" # todo
OPTS+=" --openai_url YOUR_OPENAI_URL" # todo
OPTS+=" --use_vllm True"
OPTS+=" --kb_path $kb_path"
OPTS+=" --kb_emb_path $kb_emb_path"
OPTS+=" --kb_sentence_path $kb_sentence_path"
# eval_rag
OPTS+=" --eval_prompt_path prompt/eval.json"

CMD="python3 ./eval.py ${OPTS}"
echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD