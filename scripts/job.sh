./scripts/generate_instructions.sh 4000 5 data/job.jsonl job_instruction.jsonl http://127.0.0.1:11023/v1/
./scripts/is_clf_or_not.sh 5 job_instruction.jsonl job_cls.jsonl http://127.0.0.1:11023/v1/
./scripts/generate_instances.sh job_instruction.jsonl job_instances.jsonl job_cls.jsonl 5 http://127.0.0.1:11023/v1/
./scripts/prepare_for_finetuning.sh job_instances.jsonl job_cls.jsonl job job.jsonl