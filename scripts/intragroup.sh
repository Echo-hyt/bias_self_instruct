./scripts/generate_instructions.sh 4000 5 data/intragroup.jsonl intragroup_instruction.jsonl http://127.0.0.1:11026/v1/
./scripts/is_clf_or_not.sh 5 intragroup_instruction.jsonl intragroup_cls.jsonl http://127.0.0.1:11026/v1/
./scripts/generate_instances.sh intragroup_instruction.jsonl intragroup_instances.jsonl intragroup_cls.jsonl 5 http://127.0.0.1:11026/v1/
./scripts/prepare_for_finetuning.sh intragroup_instances.jsonl intragroup_cls.jsonl intragroup intragroup.jsonl
python ./self_instruct/label.py --input_file data/gpt3_generations/intragroup/all_generated_instances.jsonl --output_instruction data/temp/intragroup_ins.jsonl --output_content data/temp/intragroup_cont.jsonl --output_instruction_text data/temp/intragroup_ins.txt --output_content_text data/temp/intragroup_cont.txt --output_bias data/label_data/intragroup_bias.jsonl --output_non_bias data/label_data/intragroup_non_bias.jsonl &
