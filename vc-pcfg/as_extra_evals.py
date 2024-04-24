import os
from pathlib import Path
import csv, json
import pandas as pd
from ast import literal_eval
import numpy as np
import argparse, logging
#from collections import defaultdict

import torch
#from torch_struct import SentCFG

from vpcfg.as_dataloader import get_data_iters, set_constant
from vpcfg.utils import Vocabulary, save_checkpoint
#from vpcfg.as_evaluation import AverageMeter, LogCollector, semantic_bootstrapping_test, syntactic_bootstrapping_test
from vpcfg.as_vocab import get_vocab


parser = argparse.ArgumentParser()
#parser.add_argument('--tree_file', default='../../../runs/out-dist/syn-first-model/91/sem/29.csv', type=str, help='')
#parser.add_argument('--tree_f1_file', default='../../../runs/out-dist/f1_gold-parse.csv', type=str, help='')

parser.add_argument('--data_path', default='../preprocessed-data/abstractscenes', help='path to datasets')
parser.add_argument('--logger_name', default='../../../scratch/vcpcfg/parses', help='location for model outputs and logfiles to be saved')
parser.add_argument('--model_init', default='../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s1018/checkpoints/29.pth.tar', type=str, help='checkpoint to initialize model with')
parser.add_argument('--tiny', action='store_true', help='if testing will create tiny dataloaders')
parser.add_argument('--log_step', default=500, type=int, help='number of steps to print and record the log')

parser.add_argument('--out_file', default='pred-parse.json')
# Inference options


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
            
##################################
def main_get_trees_with_cat(opt):
    #initialize logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    else:
        print('Creating {}'.format(opt.logger_name))
        os.mkdir(opt.logger_name)
        os.mkdir(opt.logger_name+'/parses')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(opt.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    # load checkpoint
    checkpoint = torch.load(opt.model_init, map_location='cpu')
    model_opt = checkpoint['opt']
    vocab = get_vocab(opt.data_path)
    model_opt.vocab_size = len(vocab)
    # construct the model
    if not model_opt.visual_mode:
        from vpcfg.model import VGCPCFGs 
    else:
        from vpcfg.model_vis import VGCPCFGs 
        model = VGCPCFGs(model_opt, vocab, logger)
    parser_params = checkpoint['model']
    model.set_state_dict(parser_params)
    model.eval()
    #model.to('cuda')
    
    # Load data loaders
    set_constant(model_opt.visual_mode, model_opt.max_length)
    _, _, sem_test_loader = get_data_iters(opt.data_path, model_opt.prefix, vocab, model_opt.batch_size, model_opt.workers, load_img=model_opt.visual_mode, encoder_file=model_opt.encoder_file, img_dim=model_opt.img_dim, shuffle=model_opt.shuffle, sampler=None, tiny=opt.tiny, one_shot=model_opt.one_shot)
    logger.info("Number of semantic test items: {}".format(sem_test_loader.dataset.length))
    nbatch = len(sem_test_loader)
    pred_cats = dict()
    logger.info("Starting parsing: ")
    for i, (images, captions, lengths, ids, spans) in enumerate(sem_test_loader):
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()
        bsize = captions.size(0) 
        nll, kl, span_margs, argmax_spans, trees, lprobs = model.forward_parser(captions, lengths)
        for b in range(bsize):
            cats = [(a[0], a[2]) for a in argmax_spans[b] if a[0] == a[1]]
            ordered_cats = sorted(cats, key=lambda cat: cat[0])
            pred_cats[ids[b]] = {'spans':argmax_spans[b], 'cats':ordered_cats}
        if i % opt.log_step == 0:
            logger.info('[{0}/{1}]'.format(i, nbatch))
    logger_dir = Path(opt.logger_name)
    outfile = logger_dir / opt.out_file
    with open(outfile, 'w') as fw:
        json.dump(pred_cats, fw)
    
def main_create_contingency_tables(opt):
    #get gold categories
    gold_file = Path(opt.data_path) / "all_gold_caps.json"
    with open(gold_file, "r") as f:
        gold_data = f.readlines()
    #get parses
    log_dir = Path(opt.logger_name)
    parse_file = log_dir / opt.out_file
    with open(parse_file, "r") as f:
        parse_data = json.load(f)
    #for parse in parses
    sent_ids = []
    word_idxs = []
    pred_cats = []
    gold_cats = []
    captions = []
    for sent_id in parse_data.keys():
        pcats = parse_data[sent_id]['cats']
        gold_item = json.loads(gold_data[int(sent_id)])
        gcats = gold_item[3]
        cap = gold_item[0]
        for (idx, pcat) in pcats:
            gcat = gcats[idx]
            pcat = 'C'+str(pcat)
            sent_ids.append(sent_id)
            word_idxs.append(idx)
            pred_cats.append(pcat)
            gold_cats.append(gcat)
            captions.append(cap)
    df_pairs = pd.DataFrame({'sent_id': sent_ids, 'caption': captions, 'word_idx': word_idxs, 'pred_cat': pred_cats, 'gold_cat': gold_cats})
    contingency_table = pd.crosstab(index=df_pairs['pred_cat'], columns=df_pairs['gold_cat'], margins=True)
    df_file = opt.out_file+'_df.csv'
    ct_file = opt.out_file+'_ct.csv'
    df_path = log_dir / df_file
    ct_path = log_dir / ct_file
    df_pairs.to_csv(str(df_path), index=False)
    contingency_table.to_csv(str(ct_path))

if __name__ == '__main__':
    #main_left_right_tree_branches(opt)
    main_get_trees_with_cat(opt)
    main_create_contingency_tables(opt)
