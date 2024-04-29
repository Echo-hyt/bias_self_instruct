./scripts/generate_instructions.sh 4000 5 data/regional.jsonl regional_instruction.jsonl http://127.0.0.1:11009/v1/
./scripts/is_clf_or_not.sh 5 regional_instruction.jsonl regional_cls.jsonl http://127.0.0.1:11009/v1/
./scripts/generate_instances.sh regional_instruction.jsonl regional_instances.jsonl regional_cls.jsonl 5 http://127.0.0.1:11009/v1/
./scripts/prepare_for_finetuning.sh regional_instances.jsonl regional_cls.jsonl regional regional.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/regional/all_generated_instances.jsonl --output_instruction data/temp/regional_ins.jsonl --output_content data/temp/regional_cont.jsonl --output_instruction_text data/temp/regional_ins.txt --output_content_text data/temp/regional_cont.txt --output_bias data/label_data/regional_bias.jsonl --output_non_bias data/label_data/regional_non_bias.jsonl &
