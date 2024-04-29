./scripts/generate_instructions.sh 4000 5 data/personality.jsonl personality_instruction.jsonl http://127.0.0.1:11013/v1/
./scripts/is_clf_or_not.sh 5 personality_instruction.jsonl personality_cls.jsonl http://127.0.0.1:11013/v1/
./scripts/generate_instances.sh personality_instruction.jsonl personality_instances.jsonl personality_cls.jsonl 5 http://127.0.0.1:11013/v1/
./scripts/prepare_for_finetuning.sh personality_instances.jsonl personality_cls.jsonl personality personality.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/personality/all_generated_instances.jsonl --output_instruction data/temp/personality_ins.jsonl --output_content data/temp/personality_cont.jsonl --output_instruction_text data/temp/personality_ins.txt --output_content_text data/temp/personality_cont.txt --output_bias data/label_data/personality_bias.jsonl --output_non_bias data/label_data/personality_non_bias.jsonl &
