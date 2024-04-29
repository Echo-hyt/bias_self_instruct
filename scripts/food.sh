./scripts/generate_instructions.sh 4000 5 data/food.jsonl food_instruction.jsonl http://127.0.0.1:11003/v1/
./scripts/is_clf_or_not.sh 5 food_instruction.jsonl food_cls.jsonl http://127.0.0.1:11003/v1/
./scripts/generate_instances.sh food_instruction.jsonl food_instances.jsonl food_cls.jsonl 5 http://127.0.0.1:11003/v1/
./scripts/prepare_for_finetuning.sh food_instances.jsonl food_cls.jsonl food food.jsonl