# todo
# source xxx/miniconda3/etc/profile.d/conda.sh

SCRIPT_ABS_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_ABS_PATH")

# todo
LOG_FILE="$SCRIPT_DIR/logs/YOUR_TIME_LOG_PATH.log"
ERROR_LOG_FILE="$SCRIPT_DIR/logs/YOUR_LOG_PATH.log"
ChatGLM3_data_format_dir="data/ChatGLM3/YOUR_PATH_TO_ChatGLM3_data_format_dir"
Finetune_file_name="YOUR_FINETUNE_FILE_NAME"

# todo: this is example path format
model_save_path="$SCRIPT_DIR/model/llama_finetuned_model"
kb_path="data/loogle/KB/KB_{id}.json"
kb_emb_path="data/loogle/embeddings/{bge_type}/sentence_embeddings_{id}_{chunk_size}.pkl"
kb_sentence_path="data/loogle/sentences/sentence_sentences_{id}_{chunk_size}.pkl"


# todo
total_CUDA=0,1,2,3
CUDA_1=0,1
CUDA_2=2,3
MASTER_PORT=12245

log_time() {
    local cmd_description="$1"
    shift
    local cmd="$@"

    local start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "START: [$cmd_description] at $start_time" >> "$LOG_FILE"

    { time bash -c "$cmd"; } 2>> "$ERROR_LOG_FILE"

    local end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "END: [$cmd_description] at $end_time" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
}

# todo: you can give iter_num, train_step_num and split_num directly or give total_train_step, KB_size and to calculate others.
# total_train_step= # todo: annotation_data/(batch_size*gradient_accumulation_steps*cuda_num)
# pairs_w_tokens=
# KB_size=
# iter_num=3
# # 32=batch_size*gradient_accumulation_steps*cuda_num
# train_step_num=$(echo "scale=0; $KB_size/32 * $pair_w_tokens / $iter_num" | bc)
# split_num=$((total_train_step / train_step_num))

iter_num=1
train_step_num=1
split_num=1

# 分割数据集
cd verify
OPTS=""
OPTS+=" --in_path ${ChatGLM3_data_format_dir}/YOUR_DATASET_PATH.json" # todo
OPTS+=" --output_dir ${ChatGLM3_data_format_dir}/${split_num}_${train_step_num}"
OPTS+=" --num $split_num"
CMD="python split_verify.py  ${OPTS}"

echo "split dataset..."
log_time "${split_num}_${train_step_num} llama_split.py" "$CMD"

# 分割的数量循环
for ((step_num=1; step_num<=$iter_num; step_num+=1)); do
    conda activate xxx # todo KBAda environment
    export CUDA_VISIBLE_DEVICES=$total_CUDA
    cd ../finetune
    OPTS=""
    OPTS+=" --key_word loogle" # todo: dataset key(finetune/LLaMA-Factory/data/dataset_info.json)
    if [ "$step_num" -eq 1 ]; then
        OPTS+=" --filename ${ChatGLM3_data_format_dir}/${split_num}_${train_step_num}/part${step_num}.json"
    else
        OPTS+=" --filename ${ChatGLM3_data_format_dir}/${split_num}_${train_step_num}/part${step_num}_verify.json"
    fi
    CMD="python change_info.py ${OPTS}"
    echo "change info..."
    echo "${CMD}"

    log_time "${split_num}_${train_step_num} change_info.py" "$CMD" &
    
    wait

    cd ../finetune/LLaMA-Factory
    conda activate xxx # todo llama_factory environment

    last_num=$((step_num - 1))
    next_num=$((step_num + 1))
    now_step=$((train_step_num * step_num))
    last_step=$((train_step_num * last_num))
    echo "current path"
    pwd
    new_output_dir="${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${step_num}"
    new_save_steps="${train_step_num}"
    new_dataset="loogle" # todo: dataset key(finetune/LLaMA-Factory/data/dataset_info.json)
    checkpoint_path="${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${last_num}/checkpoint-${last_step}"
    if [ "$step_num" -ne 1 ]; then
        if grep -q "^resume_from_checkpoint:" config/llama3_lora_sft.yaml; then
            sed -i "s|^resume_from_checkpoint:.*|resume_from_checkpoint: $checkpoint_path|" config/llama3_lora_sft.yaml
        else
            printf "\nresume_from_checkpoint: $checkpoint_path\n" >>  config/llama3_lora_sft.yaml
        fi
    else
        sed -i "/^resume_from_checkpoint:.*/d" config/llama3_lora_sft.yaml
    fi
    sed -i "s|^output_dir:.*|output_dir: $new_output_dir|" config/llama3_lora_sft.yaml
    sed -i "s|^save_steps:.*|save_steps: $new_save_steps|" config/llama3_lora_sft.yaml
    sed -i "s|^dataset:.*|dataset: $new_dataset|" config/llama3_lora_sft.yaml

    llamafactory-cli train config/llama3_lora_sft.yaml

    # rm -r "${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${last_num}/v1-${last_step}"
    echo "-------export start------"

    adapter_name_or_path="${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${step_num}/checkpoint-${now_step}"
    export_dir="${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${step_num}/v1-${now_step}" 

    sed -i "s|^adapter_name_or_path:.*|adapter_name_or_path: $adapter_name_or_path|" config/llama3_merge_lora.yaml
    sed -i "s|^export_dir:.*|export_dir: $export_dir|" config/llama3_merge_lora.yaml
    llamafactory-cli export config/llama3_merge_lora.yaml &
    wait

    conda activate xxx # todo KBAda environment
    if [ "$step_num" -eq $iter_num ]; then
        echo "next"
    else
        export CUDA_VISIBLE_DEVICES=$CUDA_1
        cd ../../verify
        OPTS=""
        OPTS+=" --kb_path $kb_path"
        OPTS+=" --kb_emb_path $kb_emb_path"
        OPTS+=" --kb_sentence_path $kb_sentence_path"

        OPTS+=" --in_path ${ChatGLM3_data_format_dir}/${split_num}_${train_step_num}/part${next_num}.json"
        OPTS+=" --out_path ${ChatGLM3_data_format_dir}/${split_num}_${train_step_num}/part${next_num}_verify.json"
        OPTS+=" --model_path ${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${step_num}/v1-${now_step}"
        OPTS+=" --language English"
        is_llama="true"
        if [ "${is_llama}" == "true" ]; then
            OPTS+=" --is_llama"
        fi
        CMD="python3 ./gen_iterate_verify.py ${OPTS}"
        echo "-------final CMD is------"
        echo "${CMD}"
        echo "-------final CMD end------"
        log_time "gen_iterate_verify.py (iter $step_num)" "$CMD" &
    fi
    wait
    # rm -r "${model_save_path}/${Finetune_file_name}_${split_num}_${train_step_num}/iter${step_num}/v1-${train_step_num}"
done
