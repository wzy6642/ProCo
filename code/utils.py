# -*- coding: utf-8 -*-
import sys


import tiktoken
import openai


# configuration
data_name_choices = ['NQ', 'TriviaQA', 'WebQ']
data_save_path = {
    'NQ': r'E:\research\08_QA-verify-rectify\datasets\Natural Questions\nq-test.jsonl', 
    'TriviaQA': r'E:\research\08_QA-verify-rectify\datasets\TriviaQA\tqa-test.jsonl', 
    'WebQ': r'E:\research\08_QA-verify-rectify\datasets\WebQuestions\webq-test.jsonl'
}
openai.api_key = "***" 
openai.api_base = "***"


def load_txt_data(path):
    with open(path, 'r', encoding='gb18030', errors='ignore') as f:
        data = f.readlines()
    eval_data = []
    for sub_data in data:
        try:
            sub_data = eval(sub_data)
            eval_data.append(sub_data)
        except Exception as e:
            eval_data.append({
                'question': 'None', 
                'gold_answer': ['None'], 
                'final_answer': 'NoneAnswer'
            })
    return eval_data


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def check_string(s):
    if s == "" or s == "IP访问频率过高,请稍后再试":
        raise ValueError("Empty string encountered.")


def answer_by_gpt_3_5_turbo(prompt, model, max_length):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=[{"role": "user", "content": f"{prompt}"}], max_tokens=max_length, temperature=0.7)
    response = completion.choices[0].message.content
    try:
        check_string(response)
    except Exception as e:
        print(e) 
        sys.exit(1)
    return response

