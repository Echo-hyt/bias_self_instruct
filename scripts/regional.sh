./scripts/generate_instructions.sh 4000 5 data/regional.jsonl regional_instruction.jsonl http://127.0.0.1:11009/v1/
./scripts/is_clf_or_not.sh 5 regional_instruction.jsonl regional_cls.jsonl http://127.0.0.1:11009/v1/
./scripts/generate_instances.sh regional_instruction.jsonl regional_instances.jsonl regional_cls.jsonl 5 http://127.0.0.1:11009/v1/
./scripts/prepare_for_finetuning.sh regional_instances.jsonl regional_cls.jsonl regional regional.jsonl