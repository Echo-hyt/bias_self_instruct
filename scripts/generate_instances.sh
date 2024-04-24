batch_dir=data/gpt3_generations/
input_file=$1
output_file=$2
cls_file=$3
request_batch_size=$4
url=$5


python self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --engine davinci \
    --max_instances_to_gen 2 \
    --cls_file ${cls_file} \
    --request_batch_size ${request_batch_size} \
    --url ${url}
