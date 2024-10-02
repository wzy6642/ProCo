# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import itertools
from copy import deepcopy

from tqdm import tqdm
import tiktoken

from utils import data_name_choices
import utils


parser = argparse.ArgumentParser(description="index of datasets")
parser.add_argument('--data_index', type=int, required=True, metavar='', default=0, help="0: 'Natural Questions', 1: 'TriviaQA', 2: 'WebQuestions'")
args = parser.parse_args()


## configure
prompt_strategy = 'GenRead-ProCo'
backbone_language_model = "GPT-3.5-Turbo"
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)
data_name = data_name_choices[args.data_index]
result_path = os.path.join('../result/', f'{data_name.capitalize()}-{prompt_strategy.capitalize()}-{backbone_language_model.capitalize()}.txt')
print(f'\n==============================={data_name.capitalize()}-{prompt_strategy.capitalize()}-{backbone_language_model.capitalize()}=====================================')

results = utils.load_txt_data(result_path)[:2000]

question = [result.get('question') for result in results]
gold = [result.get('gold_answer') for result in results]


def get_prediction(result, keys):
    try:
        if len(keys)==2:
            return result.get(keys[0]).get(keys[1])
        else:
            return result.get(keys[0]).get(keys[1]).get(keys[2])
    except Exception as e:
        return result.get('final_answer')
    
final_pred = [result.get('final_answer') for result in results]


def word_combinations(sentence):
    words = sentence.split(' ')
    combinations = []
    for subset in itertools.permutations(words, len(words)):
        combinations.append(' '.join(subset))
    return combinations


def EM_Score(pred, gold, pred_refine, gold_refine):
    count = 0
    correct = []
    for index in tqdm(range(len(gold))):
        if len(gold[index])==1:
            if gold[index][0].replace("``", "").replace("''", "").replace(" ", "").lower() in pred[index].replace("``", "").replace("''", "").replace(" ", "").lower():
                count += 1
                correct.append(index)
                pred_refine[index] = gold[index][0]
        else:
            for sub_answer in [item.replace("``", "").replace("''", "").replace(" ", "").lower() for item in gold[index]]:
                if sub_answer in pred[index].replace("``", "").replace("''", "").replace(" ", "").lower():
                    count += 1
                    correct.append(index)
                    pred_refine[index] = sub_answer
                    gold_refine[index] = [sub_answer]
                    break
                elif pred[index].replace("``", "").replace("''", "").replace(" ", "").lower() in sub_answer:
                    count += 1
                    correct.append(index)
                    pred_refine[index] = sub_answer
                    gold_refine[index] = [sub_answer]
                    break
    return count, correct, pred_refine, gold_refine


def calculate_f1_score(string1, string2):
    # 将字符串转换为集合
    set1 = set(encoding.encode(string1.lower()))
    set2 = set(encoding.encode(string2.lower()))
    
    # 计算准确率
    precision = len(set1.intersection(set2)) / len(set1)
    
    # 计算召回率
    recall = len(set1.intersection(set2)) / len(set2)
    
    # 计算F1分数
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

count, correct, pred_refine, gold_refine = EM_Score(final_pred, gold, deepcopy(final_pred), deepcopy(gold))
print(f'Exact Match Score of {data_name.capitalize()}-{prompt_strategy.capitalize()}-{backbone_language_model.capitalize()} final: {count / len(gold)*100:.2f}%\n')
print(f"F1 Score of {data_name.capitalize()}-{prompt_strategy.capitalize()}-{backbone_language_model.capitalize()}: {sum([calculate_f1_score(pred_refine[i], ' '.join(gold_refine[i])) for i in range(len(gold_refine))]) / len(gold_refine)*100:.2f}%\n")

