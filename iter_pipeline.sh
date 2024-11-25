SCRIPT_ABS_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_ABS_PATH")

# todo
LOG_FILE="$SCRIPT_DIR/logs/YOUR_TIME_LOG_PATH.log"
ERROR_LOG_FILE="$SCRIPT_DIR/logs/YOUR_LOG_PATH.log"
bm_data_format_dir="$SCRIPT_DIR/data/bm/YOUR_PATH_TO_bm_data_format_dir"
Finetune_file_name="YOUR_FINETUNE_FILE_NAME"
model_name_or_path="YOUR_PATH_TO_LLM"

# todo: this is example path format
model_save_path="$SCRIPT_DIR/model/cpm_finetuned_model"
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
# total_train_step= # annotation_data/(batch_size*gradient_accumulation_steps*cuda_num)
# pair_w_tokens=
# KB_size=
# iter_num=3
# # 8=batch_size*gradient_accumulation_steps
# train_step_num=$(echo "scale=0; $KB_size/8 * $pair_w_tokens / $iter_num" | bc)
# split_num=$((total_train_step / train_step_num))

iter_num=1
train_step_num=1
split_num=1

# 分割数据集
cd verify
OPTS=""
OPTS+=" --in_path ${bm_data_format_dir}/YOUR_DATASET_PATH.jsonl" # todo
OPTS+=" --output_dir ${bm_data_format_dir}/${split_num}_${train_step_num}"
OPTS+=" --num $split_num"
CMD="python split_verify.py  ${OPTS}"

echo "split dataset..."
log_time "${split_num}_${train_step_num} split_verify.py" "$CMD"

# 分割的数量循环
for ((step_num=1; step_num<=$iter_num; step_num+=1)); do
    export CUDA_VISIBLE_DEVICES=$total_CUDA
    cd ../finetune
    # 第一轮用原始的训练
    last_num=$((step_num - 1))
    next_num=$((step_num + 1))
    echo "current path"
    pwd
    MASTER_ADDR=localhost
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=4
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                    --nnodes $NNODES \
                    --node_rank $NODE_RANK \
                    --master_addr $MASTER_ADDR \
                    --master_port $MASTER_PORT"

    OPTS=""
    OPTS+=" --gradient_accumulation_steps 4" # todo
    OPTS+=" --logging_step 5" # todo
    OPTS+=" --batch_size_per_device 2" # todo
    OPTS+=" --save_step $train_step_num"
    OPTS+=" --epochs 1" # todo
    # OPTS+=" --seed 42" # todo
    OPTS+=" --lr 1e-5" # todo
    OPTS+=" --max_seq_length 3000" # todo
    OPTS+=" --weight-decay 0.1" # todo
    if [ "$step_num" -eq 1 ]; then
        OPTS+=" --warmup_iters 50" # todo
    else
        OPTS+=" --warmup_iters 0" # todo
    fi
    OPTS+=" --lr-decay-iters $(($iter_num * $train_step_num / 4))" # todo
    # OPTS+=" --lr-decay-iters $(($train_step_num))"
    OPTS+=" --lr-decay-style cosine" # todo
    OPTS+=" --start-step 0"
    OPTS+=" --model_name_or_path $YOUR_PATH_TO_LLM"
    OPTS+=" --tensorboard ${model_save_path}/$train_step_num/iter$step_num/logs"
    OPTS+=" --model mc_gold_v1"
    OPTS+=" --save_dir ${model_save_path}/$train_step_num/iter$step_num"
    OPTS+=" --data_dir ${bm_data_format_dir}/${split_num}_${train_step_num}"
    if [ "$step_num" -eq 1 ]; then
        OPTS+=" --ultra_split part$step_num"
    else
        OPTS+=" --load_ckpt ${model_save_path}/$train_step_num/iter$last_num/mc_gold_v1/step_$train_step_num" 
        OPTS+=" --ultra_split part${step_num}_verify"
    fi
    OPTS+=" --save_limit 5"
    OPTS+=" --loss-scale 32768000"
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
    CMD="torchrun ${DISTRIBUTED_ARGS} mc_finetune/src/train_bm_modelcenter.py ${OPTS}"
    echo "Starting training..."
    echo $CMD
    log_time "train_bm_modelcenter.py (iter $step_num)" "$CMD"

    # tranform the model
    echo "current path"
    pwd
    OPTS=""
    OPTS+=" --inpath ${model_save_path}/$train_step_num/iter$step_num/mc_gold_v1/step_$train_step_num"
    OPTS+=" --inpath2 $YOUR_PATH_TO_LLM"
    OPTS+=" --outpath ${model_save_path}/$train_step_num/iter$step_num/mc/v1_$train_step_num"
    OPTS+=" --has_special false"
    CMD="python3 ./bmtrainMiniCPM_hugMiniCPM.py ${OPTS}"
    echo "-------final CMD is------"
    echo "${CMD}"
    echo "-------final CMD end------"
    log_time "bmtrainMiniCPM_hugMiniCPM.py (iter $step_num)" "$CMD" &
    wait

    # verify
    if [ "$step_num" -eq $iter_num ]; then
        echo "next"
    else
        export CUDA_VISIBLE_DEVICES=$CUDA_1
        cd ../verify
        OPTS=""
        OPTS+=" --kb_path $kb_path"
        OPTS+=" --kb_emb_path $kb_emb_path"
        OPTS+=" --kb_sentence_path $kb_sentence_path"
        OPTS+=" --in_path ${bm_data_format_dir}/${split_num}_${train_step_num}/part${next_num}.jsonl"
        OPTS+=" --out_path ${bm_data_format_dir}/${split_num}_${train_step_num}/part${next_num}_verify.jsonl"
        OPTS+=" --model_path ${model_save_path}/$train_step_num/iter$step_num/mc/v1_$train_step_num"
        OPTS+=" --language English" # todo
        CMD="python3 ./gen_iterate_verify.py ${OPTS}"
        echo "-------final CMD is------"
        echo "${CMD}"
        echo "-------final CMD end------"
        log_time "gen_iterate_verify.py (iter $step_num)" "$CMD" &
    fi

    wait
    # rm -r "${model_save_path}/$train_step_num/iter$step_num/mc/v1_$train_step_num"
done
