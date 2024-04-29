./scripts/generate_instructions.sh 4000 5 data/food.jsonl food_instruction.jsonl http://127.0.0.1:11003/v1/
./scripts/is_clf_or_not.sh 5 food_instruction.jsonl food_cls.jsonl http://127.0.0.1:11003/v1/
./scripts/generate_instances.sh food_instruction.jsonl food_instances.jsonl food_cls.jsonl 5 http://127.0.0.1:11003/v1/
./scripts/prepare_for_finetuning.sh food_instances.jsonl food_cls.jsonl food food.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/food/all_generated_instances.jsonl --output_instruction data/temp/food_ins.jsonl --output_content data/temp/food_cont.jsonl --output_instruction_text data/temp/food_ins.txt --output_content_text data/temp/food_cont.txt --output_bias data/label_data/food_bias.jsonl --output_non_bias data/label_data/food_non_bias.jsonl &
