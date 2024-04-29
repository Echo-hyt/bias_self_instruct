./scripts/generate_instructions.sh 4000 5 data/national.jsonl national_instruction.jsonl http://127.0.0.1:11016/v1/
./scripts/is_clf_or_not.sh 5 national_instruction.jsonl national_cls.jsonl http://127.0.0.1:11016/v1/
./scripts/generate_instances.sh national_instruction.jsonl national_instances.jsonl national_cls.jsonl 5 http://127.0.0.1:11016/v1/
./scripts/prepare_for_finetuning.sh national_instances.jsonl national_cls.jsonl national national.jsonl