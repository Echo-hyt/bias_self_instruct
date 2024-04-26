batch_dir=data/gpt3_generations/
instance_files=$1
classification_type_files=$2
output_dir=$3
seed_tasks_path=$4


python self_instruct/prepare_for_finetuning.py \
    --instance_files ${batch_dir}/${instance_files} \
    --classification_type_files ${batch_dir}/${classification_type_files} \
    --output_dir ${batch_dir}/${output_dir} \
    --include_seed_tasks \
    --seed_tasks_path data/${seed_tasks_path} \