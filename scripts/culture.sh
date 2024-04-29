./scripts/generate_instructions.sh 4000 5 data/culture.jsonl culture_instruction.jsonl http://127.0.0.1:11006/v1/
./scripts/is_clf_or_not.sh 5 culture_instruction.jsonl culture_cls.jsonl http://127.0.0.1:11006/v1/
./scripts/generate_instances.sh culture_instruction.jsonl culture_instances.jsonl culture_cls.jsonl 5 http://127.0.0.1:11006/v1/
./scripts/prepare_for_finetuning.sh culture_instances.jsonl culture_cls.jsonl culture culture.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/culture/all_generated_instances.jsonl --output_instruction data/temp/culture_ins.jsonl --output_content data/temp/culture_cont.jsonl --output_instruction_text data/temp/culture_ins.txt --output_content_text data/temp/culture_cont.txt --output_bias data/label_data/culture_bias.jsonl --output_non_bias data/label_data/culture_non_bias.jsonl &
