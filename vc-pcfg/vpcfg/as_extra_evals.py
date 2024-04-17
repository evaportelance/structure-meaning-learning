import os, re
#from pathlib import Path
import csv
from ast import literal_eval
#import numpy as np
import argparse
#from collections import defaultdict

#import torch
#from torch_struct import SentCFG

#from . import data

#from .utils import Vocabulary
#from .module import CompoundCFG


parser = argparse.ArgumentParser()
parser.add_argument('--tree_file', default='../../../runs/out-dist/syn-first-model/91/sem/29.csv', type=str, help='')
parser.add_argument('--tree_f1_file', default='../../../runs/out-dist/f1_gold-parse.csv', type=str, help='')
#parser.add_argument('--model_file', default='../../preprocessed-data/abstractscenes', type=str, help='')

opt = parser.parse_args()

def get_f1(span1, span2):
    overlap = span1.intersection(span2)
    prec = float(len(overlap)) / (len(span1) + 1e-8)
    reca = float(len(overlap)) / (len(span2) + 1e-8)
    if len(span2) == 0:
        reca = 1.
        if len(span1) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1

def main_left_right_tree_branches(opt):
    right_left_spans = list()
    with open(opt.tree_file, "r") as f:
        reader = csv.reader(f, quotechar='"')
        for i, line in enumerate(reader):
            pred_span = literal_eval(line[2])
            #pred_span = literal_eval(line[1])
            gold_span = literal_eval(line[1])
            gold_f1 = line[3]
            #gold_f1 = 1.0
            pred_set = set(pred_span[:-1])
            n = len(pred_span)
            right_spans = []
            left_spans = []
            for c in range(0, n):
                right_spans.append((c, n))
                left_spans.append((0, c+1))
            right_set = set(right_spans[1:])
            left_set = set(left_spans[:-1])
            right_f1 = get_f1(pred_set, right_set)
            left_f1 = get_f1(pred_set, left_set)
            right_left_spans.append([pred_span, right_spans, left_spans, gold_span, right_f1, left_f1, gold_f1])
    with open(opt.tree_f1_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['pred_spans', 'right_spans', ' left_spans', 'gold_spans', 'right_f1', 'left_f1', 'gold_f1'])
        for row in right_left_spans:
            writer.writerow(row)


if __name__ == '__main__':
    main_left_right_tree_branches(opt)
