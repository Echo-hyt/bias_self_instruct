./scripts/generate_instructions.sh 4000 5 data/national.jsonl national_instruction.jsonl http://127.0.0.1:11016/v1/
./scripts/is_clf_or_not.sh 5 national_instruction.jsonl national_cls.jsonl http://127.0.0.1:11016/v1/
./scripts/generate_instances.sh national_instruction.jsonl national_instances.jsonl national_cls.jsonl 5 http://127.0.0.1:11016/v1/
./scripts/prepare_for_finetuning.sh national_instances.jsonl national_cls.jsonl national national.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/national/all_generated_instances.jsonl --output_instruction data/temp/national_ins.jsonl --output_content data/temp/national_cont.jsonl --output_instruction_text data/temp/national_ins.txt --output_content_text data/temp/national_cont.txt --output_bias data/label_data/national_bias.jsonl --output_non_bias data/label_data/national_non_bias.jsonl &
