import json

vocab = dict()

def add_words(ifile):
    with open(ifile, "r") as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.split("\t")[2]
            line = line.strip().split(" ")
            for word in line:
              if word in vocab:
                vocab[word] +=1
              else:
                vocab[word] = 1

def write_vocab(ofile):
    with open(ofile, "w") as fw:
        json.dump(vocab, fw)

if __name__ == '__main__':
    root_caps = "../../preprocessed-data/abstractscenes/"
    sentence_files = ["SimpleSentences1_clean.txt", "SimpleSentences2_clean.txt"]
    ofile = "complete_word_list_counts.json"

    for ifile in sentence_files:
        add_words(root_caps+ifile)

    write_vocab(root_caps+ofile)

    print(len(vocab.keys()))
