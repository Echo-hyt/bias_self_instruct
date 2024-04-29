./scripts/generate_instructions.sh 4000 5 data/religious.jsonl religious_instruction.jsonl http://127.0.0.1:11006/v1/
./scripts/is_clf_or_not.sh 5 religious_instruction.jsonl religious_cls.jsonl http://127.0.0.1:11006/v1/
./scripts/generate_instances.sh religious_instruction.jsonl religious_instances.jsonl religious_cls.jsonl 5 http://127.0.0.1:11006/v1/
./scripts/prepare_for_finetuning.sh religious_instances.jsonl religious_cls.jsonl religious religious.jsonl