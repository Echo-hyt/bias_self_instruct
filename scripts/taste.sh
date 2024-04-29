./scripts/generate_instructions.sh 4000 5 data/taste.jsonl taste_instruction.jsonl http://127.0.0.1:11003/v1/
./scripts/is_clf_or_not.sh 5 taste_instruction.jsonl taste_cls.jsonl http://127.0.0.1:11003/v1/
./scripts/generate_instances.sh taste_instruction.jsonl taste_instances.jsonl taste_cls.jsonl 5 http://127.0.0.1:11003/v1/
./scripts/prepare_for_finetuning.sh taste_instances.jsonl taste_cls.jsonl taste taste.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/taste/all_generated_instances.jsonl --output_instruction data/temp/taste_ins.jsonl --output_content data/temp/taste_cont.jsonl --output_instruction_text data/temp/taste_ins.txt --output_content_text data/temp/taste_cont.txt --output_bias data/label_data/taste_bias.jsonl --output_non_bias data/label_data/taste_non_bias.jsonl &
