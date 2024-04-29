./scripts/generate_instructions.sh 4000 5 data/intragroup.jsonl intragroup_instruction.jsonl http://127.0.0.1:11026/v1/
./scripts/is_clf_or_not.sh 5 intragroup_instruction.jsonl intragroup_cls.jsonl http://127.0.0.1:11026/v1/
./scripts/generate_instances.sh intragroup_instruction.jsonl intragroup_instances.jsonl intragroup_cls.jsonl 5 http://127.0.0.1:11026/v1/
./scripts/prepare_for_finetuning.sh intragroup_instances.jsonl intragroup_cls.jsonl intragroup intragroup.jsonl