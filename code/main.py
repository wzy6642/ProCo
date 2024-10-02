# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import json
from pprint import pprint

import numpy as np
from tqdm import tqdm

import utils, prompt
from utils import data_name_choices, data_save_path


## configure
prompt_strategy = 'GenRead-ProCo'
backbone_language_model = "GPT-3.5-Turbo"
MAX_LENGTH = 1024
encoding_name = "cl100k_base"
MAX_ITERATION = 5
num = 2000

## load data
parser = argparse.ArgumentParser(description="index of datasets")
parser.add_argument('--data_index', type=int, required=True, metavar='', default=0, help="0: 'Natural Questions', 1: 'TriviaQA', 2: 'WebQuestions'")
args = parser.parse_args()
data_name = data_name_choices[args.data_index]
save_path = os.path.join('../result/', f'{data_name.capitalize()}-{prompt_strategy.capitalize()}-{backbone_language_model.capitalize()}.txt')
with open(data_save_path.get(data_name), 'r') as f:
    decoder = json.JSONDecoder()
    data = f.readlines()
    samples = [decoder.raw_decode(i)[0] for i in data][:num]

Questions = [sample.get('question') for sample in samples]  
Answers = [sample.get('answer') for sample in samples]      
print(f'Name of dataset: {data_name}\nMean value of the question token: {np.mean([utils.num_tokens_from_string(question, encoding_name) for question in Questions])}\nNumber of questions: {len(Questions)}')


## generate answer
if not os.path.exists(save_path):
    add_idx = 0
else:
    add_idx = len(utils.load_txt_data(save_path))
for question_idx in tqdm(range(len(samples)), desc=f'{data_name} {prompt_strategy} {backbone_language_model}'):
    question_idx += add_idx
    process_record = {}
    question = Questions[question_idx]

    process_record['question'] = question
    process_record['gold_answer'] = Answers[question_idx]

    final_answer, process_record = prompt.pipline(
        process_record, 
        question.replace('?', ' ?') ,
        backbone_language_model, 
        MAX_LENGTH, 
        MAX_ITERATION
    )
    process_record['final_answer'] = final_answer
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(process_record, ensure_ascii=False) + '\n')
