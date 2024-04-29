./scripts/generate_instructions.sh 4000 5 data/taste.jsonl taste_instruction.jsonl http://127.0.0.1:11003/v1/
./scripts/is_clf_or_not.sh 5 taste_instruction.jsonl taste_cls.jsonl http://127.0.0.1:11003/v1/
./scripts/generate_instances.sh taste_instruction.jsonl taste_instances.jsonl taste_cls.jsonl 5 http://127.0.0.1:11003/v1/
./scripts/prepare_for_finetuning.sh taste_instances.jsonl taste_cls.jsonl taste taste.jsonl