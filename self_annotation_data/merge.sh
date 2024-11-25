#!/bin/bash

echo "current path"
pwd

# todo
files=("YOUR_PATH_TO_xxx.jsonl" "YOUR_PATH_TO_xxx.jsonl")  
# files=("YOUR_PATH_TO_xxx.json" "YOUR_PATH_TO_xxx.json")  
ratios=("1.0" "1.0")                     
output="YOUR_PATH_TO_merge_file"                       
shuffle="true"                     

OPTS=""
OPTS+=" --files"

for file in "${files[@]}"; do
    OPTS+=" ${file}"
done

if [ ${#ratios[@]} -gt 0 ]; then
    OPTS+=" --ratios"
    for ratio in "${ratios[@]}"; do
        OPTS+=" ${ratio}"
    done
fi

OPTS+=" --output ${output}"

if [ "${shuffle}" == "true" ]; then
    OPTS+=" --shuffle"
fi

CMD="python3 merge.py ${OPTS}"
echo "------- Final CMD is ------"
echo "${CMD}"
echo "------- Final CMD end ------"

$CMD
