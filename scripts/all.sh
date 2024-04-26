./scripts/generate_instructions.sh 2000 5 data/gender.jsonl gender_instruction.jsonl http://127.0.0.1:11003/v1/
./scripts/is_clf_or_not.sh 1 gender_instruction.jsonl gender_cls.jsonl http://127.0.0.1:11003/v1/
./scripts/generate_instances.sh gender_instruction.jsonl gender_instances.jsonl gender_cls.jsonl 1 http://127.0.0.1:11003/v1/
./scripts/prepare_for_finetuning.sh gender_instances.jsonl gender_cls.jsonl gender gender.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/gender/all_generated_instances.jsonl --output_bias $1 --output_non_bias $2