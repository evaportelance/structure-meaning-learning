import json
from pathlib import Path
import argparse
import csv
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_dir', default='../../preprocessed-data/abstractscenes', type=str, help='')
parser.add_argument('--vocab_size', default=2000, type=int, help='')

opt = parser.parse_args()
try:
    test = pos_tag(word_tokenize("This is a test."), tagset="universal")        
except:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

def get_pos_tags(captions):
    word_tag_dict = dict()
    counter = 0
    for cap in captions:
        tag_cap = pos_tag(word_tokenize(cap), tagset="universal")
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        if len(tag_cap) > 0:
            for (word, tag) in tag_cap :
                if word in word_tag_dict:
                    if tag not in word_tag_dict[word]:
                        word_tag_dict[word].append(tag)
                else:
                    word_tag_dict[word] = [tag]
    return word_tag_dict          

def main_get_verb_list_csv(opt):
    preprocessed_dir = Path(opt.preprocessed_dir)
    word_list_file = preprocessed_dir / 'complete_word_list_counts.json'
    captions_file = preprocessed_dir / 'all_caps.text'
    ofile = preprocessed_dir / 'tagged_word_list.tsv'
    with word_list_file.open("r") as f:
        word_freq_dict = json.load(f)
    with captions_file.open("r") as f:
        captions = f.readlines()
    sorted_words_freq = sorted(word_freq_dict.items(), key=lambda x:x[1], reverse=True)
    vocab = dict(sorted_words_freq[0:(opt.vocab_size+1)])
    word_tag_dict = get_pos_tags(captions)
    stemmer = SnowballStemmer("english")
    with open(ofile, 'w') as csv_file:  
        writer = csv.writer(csv_file, delimiter ='\t')
        writer.writerow(["word", "frequency", "stem", "is_verb", "tags"])
        for word in vocab.keys():
            try:
                tags = word_tag_dict[word]
            except:
                tags = []
                print(word + " " + str(vocab[word]))
            is_verb = "VERB" in tags
            freq = vocab[word]
            stem = stemmer.stem(word)
            writer.writerow([word, freq, stem, is_verb, tags])

if __name__ == '__main__':
    main_get_verb_list_csv(opt)