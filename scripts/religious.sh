./scripts/generate_instructions.sh 4000 5 data/religious.jsonl religious_instruction.jsonl http://127.0.0.1:11006/v1/
./scripts/is_clf_or_not.sh 5 religious_instruction.jsonl religious_cls.jsonl http://127.0.0.1:11006/v1/
./scripts/generate_instances.sh religious_instruction.jsonl religious_instances.jsonl religious_cls.jsonl 5 http://127.0.0.1:11006/v1/
./scripts/prepare_for_finetuning.sh religious_instances.jsonl religious_cls.jsonl religious religious.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/religious/all_generated_instances.jsonl --output_instruction data/temp/religious_ins.jsonl --output_content data/temp/religious_cont.jsonl --output_instruction_text data/temp/religious_ins.txt --output_content_text data/temp/religious_cont.txt --output_bias data/label_data/religious_bias.jsonl --output_non_bias data/label_data/religious_non_bias.jsonl &
