### Start Model 

./scripts/start_model.sh 

model path is the file path where gemma-7b is located

### Generate data

1. Generate instructions from the seed tasks

./scripts/generate_instructions.sh

2. Identify whether the instruction represents a classification task or not

./scripts/is_clf_or_not.sh

3. Generate instances for each instruction

./scripts/generate_instances.sh

4. Filtering, processing, and reformatting

./scripts/prepare_for_finetuning.sh

If you want to execute four scripts in one click 

./scripts/all.sh

Number "2000" in the first scripts is the number of data generated at one time, and number "1" in other  scripts is the batch size.

data/gender.jsonl is the seed task path