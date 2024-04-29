./scripts/generate_instructions.sh 4000 5 data/mental.jsonl mental_instruction.jsonl http://127.0.0.1:11019/v1/
./scripts/is_clf_or_not.sh 5 mental_instruction.jsonl mental_cls.jsonl http://127.0.0.1:11019/v1/
./scripts/generate_instances.sh mental_instruction.jsonl mental_instances.jsonl mental_cls.jsonl 5 http://127.0.0.1:11019/v1/
./scripts/prepare_for_finetuning.sh mental_instances.jsonl mental_cls.jsonl mental mental.jsonl