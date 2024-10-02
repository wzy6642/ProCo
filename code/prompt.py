# -*- coding: utf-8 -*-
import time
import re
from difflib import SequenceMatcher
from copy import deepcopy

import spacy
nlp = spacy.load("en_core_web_trf")
from pprint import pprint
import Levenshtein

from utils import answer_by_gpt_3_5_turbo


SHOW = True
sleep_time = 0
threshold = 0.8


def get_entity(reasoning_path):
    pattern = r'The most relevant entity is:? (.*?)and its'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result)!=0:
        entity = result[-1]
    else:
        entity = 'None'          # No match found.
    entity = entity.replace("\"", "").replace("*", "").strip()
    return entity


def get_category(reasoning_path):
    pattern = r'and its category is:? (.*?)\.'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result)!=0:
        category = result[-1]
    else:
        category = 'None'          # No match found.
    category = category.replace("\"", "").replace("*", "").strip()
    return category


def get_answer(reasoning_path):
    pattern = r'The answer is:? (.*?)\.'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result)!=0:
        answer = result[-1]
    else:
        answer = 'None'          # No match found.
    answer = answer.replace("\"", "").replace("*", "").strip()
    return answer


def get_verification_question_answer(reasoning_path):
    pattern = r'[?\"?X\"?]? refers to:? (.*?)$'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result)!=0:
        answer = result[-1]
    else:
        answer = 'None'          # No match found.
    answer = answer.replace("\"", "").replace("*", "").strip()
    if answer[-1]=='.':
        answer = answer[:-1]
    return answer


def identify_important_entity(process_record, model, max_length, question):
    state = True
    prompt = f"""Define: Named-entity recognition seeks to locate and classify named entities mentioned in a question into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. Adjectives and verbs are not entities.\n\n
                 Instruction: Given the question below, identify a list of entities (entities must appear in the problem) in the question and for each entry explain why it either is or is not an entity, and select the most relevant entity to the question from that list. The reasoning process ends with the conclusion \"The most relevant entity is (entity) and its category is (category),\" e.g., \"The most relevant entity is Notre Dame and its category is location.\"\n\n
                 Question: Which port lies between Puget Sound and Lake Washington?\n\n
                 Answer: 
                    1. Puget Sound | True | as it is a body of water (location) 
                    2. Lake Washington | True | as it is a body of water (location) 
                    3. port | False | as "port" is a generic term and not a specific entity. It could refer to various types of ports, such as maritime ports or computer ports, without further context. 
                    4. between | False | as it is not an entity. It's a preposition indicating a relationship between Puget Sound and Lake Washington. 
                    \"The most relevant entity is Lake Washington and its category is a body of water (location).\"\n\n
                 Question: {question}\n\n
                 Answer:"""
    response = answer_by_gpt_3_5_turbo(
        prompt=prompt,
        model=model,
        max_length=max_length, 
    )
    time.sleep(sleep_time)
    if SHOW:
        print(f'\n[INFO]:\t\tEntity Identification Process: {response}')
    entity = get_entity(response)
    category = get_category(response)
    if SHOW:
        print(f'\n[INFO]:\t\tEntity: {entity}, Category: {category}')
    similarity_ratio = SequenceMatcher(None, entity.lower(), question.lower()).ratio()
    if entity.lower() in question.lower():
        pass
    elif entity=='None' or category=='None' or similarity_ratio < threshold:
        try:
            doc = nlp(question)
            entity_category = [(ent.text, ent.label_) for ent in doc.ents]
            entity, category = entity_category[-1]
        except Exception as e:
            state = False
    elif entity.lower() not in question.lower():
        s = SequenceMatcher(None, entity.lower(), question.lower())
        match = s.find_longest_match(0, len(entity), 0, len(question))
        entity = entity[match.a: match.a + match.size]
    if state:
        process_record['entity_category'] = {}
        process_record['entity_category']['reasoning_path'] = response
        process_record['entity_category']['entity'] = entity
        process_record['entity_category']['category'] = category
    return entity.lower(), category.lower(), process_record, state


def construct_verification_question_pro(question, entity):
    verification_question = question.lower().replace(entity, "[X]")
    return verification_question


def generate_document(process_record, model, max_length, question, num_iter, flag):
    prompt = f"""Generate a background document to answer the given question.\n\n{question}\n\n"""
    document = answer_by_gpt_3_5_turbo(
        prompt=prompt,
        model=model,
        max_length=max_length, 
    )
    if SHOW:
        print(f'\n[INFO]:\t\tDocument: {document}')
    time.sleep(sleep_time)
    if flag=='init':
        process_record[f'{num_iter}-iter'][f'{flag}-document'] = document
    else:
        process_record[f'{num_iter}-iter']['rectification'] = {}
        process_record[f'{num_iter}-iter']['rectification'][f'{flag}-document'] = document
    return document


def generate_answer(process_record, model, max_length, question, num_iter, document, flag):
    prompt = f"""Refer to the passage below and answer the following question with just one entity.\n\nPassage: {document}\n\nQuestion: {question}\n\nThe answer is"""
    answer = answer_by_gpt_3_5_turbo(
        prompt=prompt,
        model=model,
        max_length=max_length, 
    )
    if SHOW:
        print(f'\n[INFO]:\t\tanswer: {answer}')
    time.sleep(sleep_time)
    if flag=='init':
        process_record[f'{num_iter}-iter'][f'{flag}-answer'] = answer
    else:
        process_record[f'{num_iter}-iter']['rectification'][f'{flag}-answer'] = answer
    return answer


def construct_verification_question(verification_question_pro, answer):
    verification_question = verification_question_pro + f' Suppose the answer is {answer}. What is [X]?'
    return verification_question


def solve_verification_question(process_record, model, max_length, num_iter, verification_question, category):
    prompt = f"""Answer what [X] is with just one entity.\n\n
                 Question: who wrote [X]? Suppose the answer is B. Traven. What is [X]? (The category of [X] is title of a creative work.)\n\n
                    Let's think step by step.
                    Step 1: Identify the category of the answer.
                    The answer "B. Traven" suggests that [X] is the title of a creative work, such as a book, a screenplay, or a piece of literature.
                    Step 2: Recognize the works associated with B. Traven.
                    B. Traven was a renowned author known for works such as "The Treasure of the Sierra Madre" and "The Death Ship".
                    Step 3: Determine which work matches the question "who wrote [X]?"
                    Given that B. Traven is the author, [X] must be a title of a creative work attributed to B. Traven.
                    Step 4: [X] is "The Death Ship".
                    Therefore, X refers to "The Death Ship".
                    Conclusion: [X] refers to "The Death Ship".\n\n
                 Question: {verification_question} (The category of [X] is {category})\n\n
                    Let's think step by step."""
    reasoning_path = answer_by_gpt_3_5_turbo(
        prompt=prompt,
        model=model,
        max_length=max_length, 
    )
    if SHOW:
        print(f'\n[INFO]:\t\tVerification Question Reasoning Path: {reasoning_path}')
    time.sleep(sleep_time)
    answer = get_verification_question_answer(reasoning_path)
    if SHOW:
        print(f'\n[INFO]:\t\tVerification Question Answer: {answer}')
    process_record[f'{num_iter}-iter']['verification'] = {}
    process_record[f'{num_iter}-iter']['verification']['verification_question'] = verification_question
    process_record[f'{num_iter}-iter']['verification']['verification_question_reasoning_path'] = reasoning_path
    process_record[f'{num_iter}-iter']['verification']['verification_question_answer'] = answer
    return answer


def verification_result(process_record, model, max_length, num_iter, verification_question, entity, entity_prediction):
    prompt = f"""
        Instruction: Determine whether the proposition is correct or incorrect.\n\n
        Proposition: If the answer to the question “who wrote [X]? Suppose the answer is B. Traven. What is [X]?” is “The Treasure of the Sierra Madre”, then [X] could also be “The Death Ship”\n\n
        A: Let's think step by step. The proposition states that if the answer to the question "who wrote X?" is B. Traven and X is "The Treasure of the Sierra Madre," then X could also be "The Death Ship."
        The reasoning path could be as follows:
        B. Traven is the author of "The Treasure of the Sierra Madre."
        Therefore, the proposition implies that B. Traven could also be the author of "The Death Ship."
        The result of the judgment is: The proposition is correct.\n\n
        Proposition: If the answer to the question \"{verification_question}\" is \"{entity}\", then [X] could also be \"{entity_prediction}\"\n\n
        A: Let's think step by step."""
    judgement_process = answer_by_gpt_3_5_turbo(
        prompt=prompt,
        model=model,
        max_length=max_length, 
    )
    if SHOW:
        print(f'\n[INFO]:\t\tJudgement Process: {judgement_process}')
    time.sleep(sleep_time)
    process_record[f'{num_iter}-iter']['judgement'] = {}
    process_record[f'{num_iter}-iter']['judgement']['judgement_process'] = judgement_process
    if "incorrect" in judgement_process.lower():
        process_record[f'{num_iter}-iter']['judgement']['judgement_result'] = "incorrect"
        return False
    else:
        process_record[f'{num_iter}-iter']['judgement']['judgement_result'] = "correct"
        return True


def rectified_question(question, incorrect_answer_record):
    refined_question = f"{question} (The answer is likely not in list {incorrect_answer_record})"
    return refined_question


def pipline(process_record, question, model, max_length, max_iteration):
    """pipline"""
    if SHOW:
        print(f'\n[INFO]:\t\tQuestion: {question}')
    answer_record, incorrect_answer_record = [], []
    process_record[f'0-iter'] = {}
    entity, category, process_record, state = identify_important_entity(process_record, model, max_length, question)
    verification_question_pro = construct_verification_question_pro(question, entity)
    document = generate_document(process_record, model, max_length, question, 0, 'init')
    answer = generate_answer(process_record, model, max_length, question, 0, document, 'init')
    answer_record.append(answer)
    if state:
        for num_iter in range(max_iteration):
            num_iter += 1
            process_record[f'{num_iter}-iter'] = {}
            verification_question = construct_verification_question(verification_question_pro, answer_record[-1])
            entity_prediction = solve_verification_question(process_record, model, max_length, num_iter, verification_question, category)
            judgement = verification_result(process_record, model, max_length, num_iter, verification_question, entity, entity_prediction)
            if judgement:
                break
            elif Levenshtein.distance(entity_prediction.lower(), entity.lower())<=5 or entity_prediction.lower() in entity.lower() or entity.lower() in entity_prediction.lower():
                break
            else:
                incorrect_answer_record = deepcopy(answer_record)
                refined_question = rectified_question(question, incorrect_answer_record)
                refined_document = generate_document(process_record, model, max_length, refined_question, num_iter, 'refined')
                refined_answer = generate_answer(process_record, model, max_length, refined_question, num_iter, refined_document, 'refined')
                answer_record.append(refined_answer)
            if len(answer_record)>=2:
                answer_record_history = [history.lower() for history in answer_record[:-1]]
                if answer_record[-1].lower() in answer_record_history:
                    break
    final_answer = answer_record[-1]
    return final_answer, process_record
