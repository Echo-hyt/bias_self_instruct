batch_dir=data/gpt3_generations/
request_batch_size=$1
input=$2
output=$3
url=$4



python self_instruct/identify_clf_or_not.py \
    --batch_dir ${batch_dir} \
    --request_batch_size ${request_batch_size} \
    --input ${input} \
    --engine "davinci" \
    --output ${output} \
    --url ${url}
