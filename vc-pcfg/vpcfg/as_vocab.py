import json
import pickle
import pathlib
import os

from as_word_list import get_complete_word_list
from utils import Vocabulary

def add_words(ifile, vocab):
    with open(ifile, "r") as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.split("\t")[2]
            line = line.strip().tolower().split(" ")
            for word in line:
              if word in vocab:
                vocab[word] +=1
              else:
                vocab[word] = 1
    return vocab

def write_word_list(ofile):
    with open(ofile, "w") as fw:
        json.dump(vocab, fw)

def get_complete_word_list(preprocessed_data_path):
    sentence_files = ["SimpleSentences1_clean.txt", "SimpleSentences2_clean.txt"]
    ofile = "complete_word_list_counts.json"
    vocab = dict()

    for ifile in sentence_files:
        vocab = add_words(preprocessed_data_path / ifile, vocab)

    write_word_list(preprocessed_data_path / ofile)


def create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size = 2000):
    vocab = Vocabulary()
    word_list = json.load(word_list_dir / word_list_file)
    sorted_words = sorted(word_list.items(), key=lambda x:x[1])
    id = 0
    for word,count in sorted_words:
        if id >= vocab_size:
            break
        vocab.add_word(word)
        id+=1
    pickle.dump(vocab, word_list_dir / vocab_file)
    return vocab

def get_vocab(preprocessed_dir, vocab_size = 2000):
    word_list_dir = Path(preprocessed_dir)
    word_list_file = 'complete_word_list_counts.json'
    vocab_file = 'vocab_dict.json'
    if vocab_file in os.listdir(word_list_dir):
        vocab = pickle.load(word_list_dir / vocab_file)
    elif word_list_file in os.listdir(word_list_dir):
        vocab = create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size)
    else :
        get_complete_word_list(preprocessed_dir)
        vocab = create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size)
    return vocab
