import json
from pathlib import Path
import argparse
import csv
import pandas as pd
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
    
g_selected_stems = ["push", "rescu", "teas", "argu", "hug", "warn", "feed", "meet", "fight", "invit", "drop", "open", "ride", "pour", "brought", "prepar", "toss", "use", "climb", "rais", "walk", "hide", "smile", "cheer", "laugh", "slid", "cri", "danc", "fell", "crawl"]

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

def main_get_test_item_list(opt):
    preprocessed_dir = Path(opt.preprocessed_dir)
    verb_list_file = preprocessed_dir / 'clean_verb_list.tsv'
    captions_file = preprocessed_dir / 'all_caps.text'
    out_file = preprocessed_dir / 'test_verb_ids.json'
    item_id = 0
    ids_to_stem_type = {}
    verbs_df = pd.read_csv(verb_list_file, sep='\t')
    verbs_df = verbs_df.loc[verbs_df['stem'].isin(g_selected_stems)]
    verbs = verbs_df['word'].tolist()
    verbs_dict_list = verbs_df.to_dict(orient='records')
    with open(captions_file, 'r') as f:
        for line in f.readlines():
            for verb in verbs_dict_list:
                if verb['word'] in line.split(" "):
                    ids_to_stem_type[item_id] = {'stem':verb['stem'], 'v_type':verb['type'], 'o_type':verb['object']}
                    print(verb['word'] + " : " +line)
            item_id+=1 
    print(len(ids_to_stem_type))
    with open(out_file, "w") as f:
        json.dump(ids_to_stem_type, f)
        
def main_create_id_dataframe(opt):
    preprocessed_dir = Path(opt.preprocessed_dir)
    id_json_file = preprocessed_dir / 'test_verb_ids.json'
    id_csv_file = preprocessed_dir / 'test_verb_ids.csv'
    with id_json_file.open("r") as f:
        id_dict = json.load(f)
    item_list = list()
    for idx in id_dict:
        item = {'id': idx,
               'info': id_dict[idx]}
        item_list.append(item)
    id_df = pd.json_normalize(item_list, meta=['id', ['info', 'stem'], ['info', 'v_type'], ['info','o_type']])
    id_df.columns = ['id','stem','verb_type','object_type']
    id_df.to_csv(id_csv_file, index=False) 
   
    
            
            
if __name__ == '__main__':
    # The following was used to get the list of all verbs in dataset
    #main_get_verb_list_csv(opt)
    # It was then hand cleaned and tagged for verb and object category and a curated list was selected. 
    # This selected list of verbs is used to create the test item list.
    #main_get_test_item_list(opt)
    # I also created a csv version of test item list for analysis
    main_create_id_dataframe(opt)