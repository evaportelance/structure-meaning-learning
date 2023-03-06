from datasets import load_dataset
import spacy, benepar
import torch.utils.data as data


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
            constituents0.append(example['caption_1'])
            constituents1.append(example['caption_0'])
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
