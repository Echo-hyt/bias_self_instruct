import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen


random.seed(42)

## huggingface-cli download --token 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 --resume-download meta-llama/LlamaGuard-7b --local-dir meta-llama/LlamaGuard-7b
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=1,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=1,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    parser.add_argument(
        "--cls_file",
        type=str,
        help="cls_file"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="url"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    task_clf_types = {}
    with open(os.path.join(args.batch_dir, args.cls_file)) as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]
    
    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    if task_clf_types[task["instruction"]]:
                        prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                    else:
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    # because the clf template is longer, we need to decrease the max_tokens
                    max_tokens=300 if any(task_clf_types[task["instruction"]] for task in batch) else 350,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=1.5,
                    stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization,
                    url=args.url
                    )
                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    # print(data["instance_metadata"])
                    # print('11111111111111111111111111111111111111111111111111111111111')
                    # data = {'prompt': "Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.\n\nTask: Which exercises are best for reducing belly fat at home?\nOutput:\n- Lying Leg Raises\n- Leg In And Out\n- Plank\n- Side Plank\n- Sit-ups\n\nTask: Extract all the country names in the paragraph, list them separated by commas.\nExample 1\nParagraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States.\nOutput: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.\n\nTask: Converting 85 F to Celsius.\nOutput: 85°F = 29.44°C\n\nTask: Sort the given list ascendingly. \nExample 1\nList: [10, 92, 2, 5, -4, 92, 5, 101]\nOutput: [-4, 2, 5, 5, 10, 92, 92, 101]\nExample 2\nInput 2 - List: [9.99, 10, -5, -1000, 5e6, 999]\nOutput: [-1000, -5, 9.99, 10, 999, 5e6]\n\nTask: Suggest a better and more professional rephrasing of the following sentence.\nExample 1\nSentence: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you to consider other candidates.\nOutput: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties.\nExample 2\nSentence: Just so you know, we did an experiment last week and found really surprising results - language model can improve itself!\nOutput: Our experiments last week demonstrated surprising results, proving that the language model can improve itself.\n\nTask: Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation for getting the final answer.\nExample 1\nParagraph: Gun violence in the United States results in tens of thousands of deaths and injuries annually, and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides in total, being 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides, and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm.\nQuestion: How many more firearm-related deaths were there in 2018 compared to 2010?\nOutput:\n38390 - (19392 + 11078) = 38390 - 30470 = 7920. \nSo, in 2018, there were 7920 more deaths by firearm than in 2010.\n\nTask: Write Python code to solve this leetcode problem.\nExample 1\nProblem: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.\nOutput:\nclass Solution(object):\n    def addTwoNumbers(self, l1, l2):\n        carry = 0\n        root = n = ListNode(0)\n        while l1 or l2 or carry:\n            v1 = v2 = 0\n            if l1:\n                v1 = l1.val\n                l1 = l1.next\n            if l2:\n                v2 = l2.val\n                l2 = l2.next\n            carry, val = divmod(v1+v2+carry, 10)\n            n.next = ListNode(val)\n            n = n.next\n        return root.next\n\nTask: Solve the equation and find the value of X. Show your steps.\nExample 1\nEquation: 10X + 5 = 10\nOutput: 10X = 5,  X = 0.5\nExample 2\nEquation: X + Y + 120 = 100\nOutput: X + Y = -20, X = -20 - Y\n\nTask: Write a program to compute the sum of integers from k to n.\nOutput:\ndef sum(k, n):\n    sum = 0\n    for i in range(k, n+1):\n        sum += i\n    return sum\n\nTask: Select the oldest person from the given list.\nExample 1\nList: George Washington, Confucius, Michael Jordan, Michelangelo\nOutput: Confucious\nExample 2\nList: Alan Turing, Geoffrey Hinton, Yann LeCun, Yoshua Bengio\nOutput: Alan Turing\n\nTask: Turn down a job offer by sending an email to a recruiter explaining the reason.\nOutput: Hi  [Recruiter],\nThank you so much for the generous offer to join your team. As we discussed, I’ve admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I’ve decided to accept an offer at another company.\nI would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.\nThanks again,\n[Your Name]\n\nTask: What are some of the benefits of having a mentor?\n", 'response': {'choices': [CompletionChoice(finish_reason='stop', index=0, logprobs=Logprobs(text_offset=[0, 6, 7, 13, 14, 18, 22, 30, 33, 40, 46, 47, 50, 52, 56, 62, 73, 79, 80, 83, 86, 87, 92, 103, 109, 110, 114, 118, 125, 132, 138, 139, 142, 144, 148, 158, 164, 165, 168, 171, 177, 180, 188, 193, 199, 200, 203, 205, 209, 216, 217, 220, 224, 230, 231, 235, 239, 240, 244, 249, 255, 256, 259, 261, 265, 269, 270, 277, 283, 284, 288, 292, 301, 304, 312], token_logprobs=[-0.01686622016131878, -0.004009780008345842, -0.4521952271461487, -0.541232168674469, -2.075228452682495, -0.15564166009426117, -0.15755631029605865, -0.17077404260635376, -0.3970627784729004, -0.05779021605849266, -0.0009179668850265443, -0.6078485250473022, -0.00026294111739844084, -1.0040414333343506, -0.6190113425254822, -0.14644820988178253, -0.07291557639837265, -0.00014435203047469258, -0.6294249296188354, -0.0003883084573317319, -0.9238495230674744, -0.4883309006690979, -0.000440262199845165, -0.20955032110214233, -0.00034445550409145653, -0.5626168251037598, -0.00026973424246534705, -1.6414506435394287, -0.0001722425949992612, -0.1997259110212326, -0.0021473937667906284, -0.9633272886276245, -0.0007529999129474163, -0.3506660759449005, -0.8711689114570618, -0.3474016785621643, -0.0814969539642334, -1.1921947002410889, -0.00034254882484674454, -0.7467519640922546, -0.0482247918844223, -1.0692533254623413, -0.00252404878847301, -0.540033221244812, -0.41346991062164307, -0.5062392354011536, -0.00039200251922011375, -0.22340062260627747, -0.6759859323501587, -0.0882401093840599, -0.001157567254267633, -0.0011038646334782243, -0.5541097521781921, -0.3634173572063446, -0.6814587116241455, -5.864924969500862e-05, -0.6797481179237366, -0.35456278920173645, -0.0009925207123160362, -0.5673240423202515, -0.30195003747940063, -0.6142411231994629, -0.00019095504831057042, -0.0750376433134079, -1.4065390825271606, -0.2762042284011841, -0.0003274143091402948, -1.1042966842651367, -0.564373791217804, -0.8011013269424438, -4.3987260141875595e-05, -1.3781518936157227, -0.0007339406292885542, -0.06793395429849625, -0.8758270740509033], tokens=['Output', ':', '<0x0A>', '-', 'Prov', 'ides', 'guidance', 'and', 'support', '<0x0A>', '-', 'Hel', 'ps', 'with', 'career', 'development', '<0x0A>', '-', 'Off', 'ers', 'a', 'fresh', 'perspective', '<0x0A>', '-', 'Prov', 'ides', 'account', 'ability', '<0x0A>', '-', 'Hel', 'ps', 'with', 'networking', '<0x0A>', '-', 'Off', 'ers', 'advice', 'and', 'encourag', 'ement', '<0x0A>', '-', 'Hel', 'ps', 'with', 'problem', '-', 'sol', 'ving', '<0x0A>', '-', 'Prov', 'ides', 'a', 'role', 'model', '<0x0A>', '-', 'Hel', 'ps', 'with', 'goal', '-', 'setting', '<0x0A>', '-', 'Prov', 'ides', 'construct', 'ive', 'feedback', '</s>'], top_logprobs=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]), text='Output:\n- Provides guidance and support\n- Helps with career development\n- Offers a fresh perspective\n- Provides accountability\n- Helps with networking\n- Offers advice and encouragement\n- Helps with problem-solving\n- Provides a role model\n- Helps with goal-setting\n- Provides constructive feedback')]}, 'created_at': '2024-01-24 18:24:10.791319'}
                    if results[i]["response"] is not None:
                        # print('10101010110011010101010101010101011001010110101011010110101001100')
                        response_data = results[i]["response"]["choices"][0].__dict__
                        response_data = results[i]["response"]["choices"][0].logprobs.__dict__
                        data["raw_instances"] = results[i]["response"]["choices"][0].text
                        data["instance_metadata"]["response"]["choices"][0] = response_data
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    
                    # print(data)
                    # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
