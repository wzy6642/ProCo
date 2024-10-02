# -*- coding: utf-8 -*-
import subprocess
import argparse


parser = argparse.ArgumentParser(description="index of datasets")
parser.add_argument('--data_index', type=int, required=True, metavar='', default=0, 
                    help="0: 'Natural Questions', 1: 'TriviaQA', 2: 'WebQuestions'")
args = parser.parse_args()

command = ["python", "main.py", "--data_index", str(args.data_index)]

while True:
    subprocess.run(command)
