from datasets import load_dataset
import spacy, benepar
import torch.utils.data as data
from PIL import Image
import random


class WinoDataLoader(data.Dataset):
    def __init__(self, nlp, winoground, parse_diff):
        self.nlp = nlp
        self.winoground = winoground
        self.parse_diff = parse_diff
        self.captions = list()
        self.images  = list()
        self.ids = list()
        self.trees = list()
        for i, example in enumerate(self.winoground):
            self.add(example)
        self.length = len(self.ids)

    def add(self, example):
        self.ids.append(example['id'])
        self.images.append((example['image_0'].convert("RGB"),example['image_1'].convert("RGB")))
        self.captions.append((example['caption_0'],example['caption_1']))
        parse0 = self.nlp(example['caption_0'])
        parse0 = list(parse0.sents)[0]
        constituents0 = [str(x) for x in parse0._.constituents]
        parse1 = self.nlp(example['caption_1'])
        parse1 = list(parse1.sents)[0]
        constituents1 = [str(x) for x in parse1._.constituents]
        if not self.parse_diff:
            self.trees.append((constituents0, constituents1))
        else:
            #constituents0.append(example['caption_1'])
            #constituents1.append(example['caption_0'])
            constituents0_diff = list(set(constituents0).difference(set(constituents1)))
            constituents1_diff = list(set(constituents1).difference(set(constituents0)))
            self.trees.append((constituents0_diff, constituents1_diff))

    def __getitem__(self, index):
        id = self.ids[index]
        captions = self.captions[index]
        images = self.images[index]
        trees = self.trees[index]
        return {'id':id, 'captions':captions, 'images': images, 'trees':trees}


    def __len__(self):
        return self.length

def get_winoground_data(args):
    winoground = load_dataset('facebook/winoground', use_auth_token=args.wino_token)['test']
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})
    parse_diff = args.parse_diff
    dataset = WinoDataLoader(nlp, winoground, parse_diff)
    #wino_dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    return dataset


class AbsScenesDataLoader(data.Dataset):
    def __init__(self, data_dict, nlp):
        self.img_ids = data_dict.keys()
        self.data_dict = data_dict
        self.nlp = nlp
        self.captions = list()
        self.images  = list()
        self.ids = list()
        self.trees = list()

        self.id = 0
        for img_id0, img_id1 in zip(self.img_ids[::2], self.img_ids[1::2].reverse()):
            self.add(img_id0, self.data_dict[img_id0], img_id1, self.data_dict[img_id1])
        self.length = len(self.ids)

    def add(self, img_id0, data0, img_id1, data1):
        for cap0, cap1 in zip(data0['captions'], data1['captions']):
            id = self.id
            self.ids.append(self.id)
            self.id += 1
            self.images.append((data0['img'], data1['img']))
            self.captions.append((cap0, cap1))
            parse0 = self.nlp(cap0)
            parse0 = list(parse0.sents)[0]
            constituents0 = [str(x) for x in parse0._.constituents]
            parse1 = self.nlp(cap1)
            parse1 = list(parse1.sents)[0]
            constituents1 = [str(x) for x in parse1._.constituents]
            if not self.parse_diff:
                self.trees.append((constituents0, constituents1))
            else:
                #constituents0.append(example['caption_1'])
                #constituents1.append(example['caption_0'])
                constituents0_diff = list(set(constituents0).difference(set(constituents1)))
                constituents1_diff = list(set(constituents1).difference(set(constituents0)))
                self.trees.append((constituents0_diff, constituents1_diff))

    def __getitem__(self, index):
        id = self.ids[index]
        captions = self.captions[index]
        images = self.images[index]
        trees = self.trees[index]
        return {'id':id, 'captions':captions, 'images': images, 'trees':trees}


    def __len__(self):
        return self.length


def create_abstractscenes_img_list(root):
    """ all abstractscenes images. http://optimus.cc.gatech.edu/clipart/dataset/AbstractScenes_v1.1.zip
    """
    image_list = list()
    for root, dir, files in os.walk(root):
        if len(dir) > 0:
            continue
        for fname in files:
            if fname.endswith(".png"):
                id = int(fname[:-4].replace("Scene", "").replace("_", ""))
                image_list.append((id, f"{root}/{fname}"))
    return image_list

def create_abstractscenes_caps_dict(root):
    caption_dict = dict()
    with open(f"{root}/all.id", 'r') as f1, open(f"{root}/all_caps.text", 'r') as f2:
        for id, cap in zip(f1.readlines(), f2.readlines()):
            if caption_dict[int(id)]:
                caption_dict[int(id)]['captions'].append(cap)
            else:
                caption_dict[int(id)] = {['captions':[cap]]}
    return caption_list

def create_data_split(data_dict, prop):
    def coin_flip(prop)
    train_dict() = dict()
    test_dict() = dict()
    ids = datadict.keys()
    for id in ids:
        if random.random() < prop:
            train_dict[id] = data_dict[id]
        else:
            test_dict[id] = data_dict[id]
    return train_dict, test_dict


def get_abstractscenes_data(args):
    random.seed(args.seed)
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})
    image_list = create_abstractscenes_img_list(args.as_img_dir)
    data_dict = create_abstractscenes_caps_dict(args.as_caps_dir)
    for (id, img_file) in image_list:
        img = Image.open(img_file).convert("RGB")
        data_dict[id]['img'] = img
    train_dict, test_dict = create_data_split(data_dict, args.prop)
    train_dataset = AbsScenesDataLoader(train_dict, nlp)
    test_dataset = AbsScenesDataLoader(test_dict, nlp)
    return train_dataset, test_dataset
