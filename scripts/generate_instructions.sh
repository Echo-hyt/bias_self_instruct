#!/bin/bash

# 使用参数
batch_dir=data/gpt3_generations/
num_instructions_to_generate=$1
batch_size=$2
seed_tasks_path=$3
output=$4
url=$5


python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate ${num_instructions_to_generate} \
    --batch_size ${batch_size} \
    --seed_tasks_path ${seed_tasks_path} \
    --engine "davinci" \
    --output ${output} \
    --url ${url}