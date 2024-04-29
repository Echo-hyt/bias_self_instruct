./scripts/generate_instructions.sh 4000 5 data/culture.jsonl culture_instruction.jsonl http://127.0.0.1:11006/v1/
./scripts/is_clf_or_not.sh 5 culture_instruction.jsonl culture_cls.jsonl http://127.0.0.1:11006/v1/
./scripts/generate_instances.sh culture_instruction.jsonl culture_instances.jsonl culture_cls.jsonl 5 http://127.0.0.1:11006/v1/
./scripts/prepare_for_finetuning.sh culture_instances.jsonl culture_cls.jsonl culture culture.jsonl