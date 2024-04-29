./scripts/generate_instructions.sh 4000 5 data/mental.jsonl mental_instruction.jsonl http://127.0.0.1:11019/v1/
./scripts/is_clf_or_not.sh 5 mental_instruction.jsonl mental_cls.jsonl http://127.0.0.1:11019/v1/
./scripts/generate_instances.sh mental_instruction.jsonl mental_instances.jsonl mental_cls.jsonl 5 http://127.0.0.1:11019/v1/
./scripts/prepare_for_finetuning.sh mental_instances.jsonl mental_cls.jsonl mental mental.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/mental/all_generated_instances.jsonl --output_instruction data/temp/mental_ins.jsonl --output_content data/temp/mental_cont.jsonl --output_instruction_text data/temp/mental_ins.txt --output_content_text data/temp/mental_cont.txt --output_bias data/label_data/mental_bias.jsonl --output_non_bias data/label_data/mental_non_bias.jsonl &
