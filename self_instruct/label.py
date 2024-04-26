import json
import argparse
import jsonlines
import openai
import os
from label_ins import generate_text_from_jsonl
from label_cont import generate_instance_from_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="Generated bias_jsonl file",
    )
    parser.add_argument(
        "--output_instruction",
        type=str,
        default = 'output_ins.jsonl',
        help="The instruction part of the output.",
    )
    parser.add_argument(
        "--output_content",
        type=str,
        default='output_content.jsonl',
        help="The instance part of the output.",
    )
    parser.add_argument(
        "--output_instruction_text",
        type=str,
        default='output_instruction_text',
        help="The output is a labeled text file",
    )
    parser.add_argument(
        "--output_instance_text",
        type=str,
        default='output_instance_text',
        help="The output is a labeled text file",
    )
    parser.add_argument(
        "--output_bias",
        type=str,
        help="output_bias",
    )
    parser.add_argument(
        "--output_non_bias",
        type=str,
        help="output_non_bias",
    )
    return parser.parse_args()


args = parse_args()
input = args.input_file
output_ins = args.output_instruction
output_content = args.output_content
output_instruction_text = args.output_instruction_text
output_instance_text = args.output_instance_text
output_bias = args.output_bias
output_non_bias = args.output_non_bias

def split_instruction_content(input_file, output_instruction_file, output_content_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_instruction_file, 'w', encoding='utf-8') as f_out_instruction, \
         open(output_content_file, 'w', encoding='utf-8') as f_out_content:
        
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data.get('instruction', '')
            content = data.get('input', '') + ' ' + data.get('output', '')

            # Write instruction to one file
            f_out_instruction.write(json.dumps({'instruction': instruction}, ensure_ascii=False) + '\n')
            
            # Write content to another file
            f_out_content.write(json.dumps({'content': content}, ensure_ascii=False) + '\n')

split_instruction_content(input, output_ins, output_content)

generate_text_from_jsonl(output_ins, output_instruction_text)

generate_instance_from_jsonl( output_content, output_instance_text)

# 读取两个txt文件
with open(output_instruction_text, 'r') as file1, open(output_instance_text, 'r') as file2:
    calls1 = file1.readlines()
    calls2 = file2.readlines()

# 过滤不符合条件的行
calls1 = [call.strip() for call in calls1 if call.startswith('Call') and len(call.split(',')) == 2 and call.split(',')[1].strip().isdigit() and call.split(',')[1].strip() in ['0', '1']]

# calls1 = [call.strip() for call in calls1 if call.startswith('Call') and call.split(',')[1].strip() in ['0', '1']]
calls2 = [call.strip() for call in calls2 if call.startswith('Call') and len(call.split(',')) == 2 and call.split(',')[1].strip().isdigit() and call.split(',')[1].strip() in ['0', '1']]
print(calls1)
# 创建新的bias.jsonl和nonbias.jsonl文件
bias_data = []
nonbias_data = []

# 根据call的数字对应关系将jsonl数据分别存入bias和nonbias列表中
for call1 in calls1:
  for call2 in calls2:
      # import pdb;pdb.set_trace()

      if call1[:-2] in call2 and call1[-1] == '1' and  call2[-1] == '1':
        # print('1111111111111')
        bias_data.append(call1[5:-3])

      if call1[:-2] in call2 and call1[-1] =='0' and  call2[-1] == '0':
        nonbias_data.append(call1[5:-3])
# print(bias_data)


json_data = []       

with open(input, 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 解析每行的 JSON 对象并将其添加到列表中
        json_data.append(line)


# 将bias数据写入bias.jsonl文件
with open(output_bias, 'w') as bias_file:
    for item in bias_data:
        json_data[int(item)-1] = json_data[int(item)-1][:-2] + ''', "classification": "bias"}\n'''
        bias_file.write(json_data[int(item)-1])
        

# 将nonbias数据写入nonbias.jsonl文件
with open(output_non_bias, 'w') as nonbias_file:
    for item in nonbias_data:
        json_data[int(item)-1] = json_data[int(item)-1][:-2] + ''', "classification": "non_bias"}\n'''

        nonbias_file.write(json_data[int(item)-1])
