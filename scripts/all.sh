./scripts/generate_instructions.sh 2000 data/gender.jsonl gender_instruction.jsonl http://127.0.0.1:11003/v1/
./scripts/is_clf_or_not.sh 1 gender_instruction.jsonl gender_cls.jsonl http://127.0.0.1:11003/v1/
./scripts/generate_instances.sh gender_instruction.jsonl gender_instances.jsonl gender_cls.jsonl 1 http://127.0.0.1:11003/v1/
./scripts/prepare_for_finetuning.sh gender_instances.jsonl gender_cls.jsonl gender data/gender.jsonl