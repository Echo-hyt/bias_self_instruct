import jsonlines
import openai
import os
import argparse




openai.api_key = "sk-CERvFqRCxI9hLU6uEhUqTHuLQekgbbJ7XhAl4DpgyS0nYo3B"

openai.base_url = "https://api.chatanywhere.tech/v1/"

model = 'gpt-3.5-turbo-1106'



# 读取JSONL文件，逐行生成文本并写入到目标文件中
def generate_text_from_jsonl(input_file, output_file):
    n=0
    with open(output_file, 'w') as f_out:
        with jsonlines.open(input_file) as reader:
            for line in reader:
                content = line['instruction']  # 假设JSONL文件中每行都有一个名为'content'的字段
                n +=1
                generated_text = generate_gpt(content)
                f_out.write(f"Call {n}, {generated_text}\n")

# 定义生成文本的函数
def generate_gpt(content):
    prompt = '''
            Remember, you can just output 0 or 1. Now you're a classification model which can only output 1 or 0 now. You can just output 0 or 1. You need to identify whether the input prompts you to output harmful speech related to discrimination. If yes, output 1; if no, output 0.
            Remember, you can just output 0 or 1.
                '''
    content = prompt + content
    # 你的生成文本的逻辑
    # 例如，你可以调用OpenAI API来生成文本
    # 这里只是一个示例
    # 获取模型的响应，并从中提取消息内
    # completion = openai.completions.create(model=model, prompt=prompt, max_tokens=1024)
    completion = openai.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": content}],
      max_tokens= 5
    )

    print(completion)
    message = completion.choices[0].message.content
    print(completion.choices[0].message.content)
    # message = completion.choices[0].message["content"]
    return message

