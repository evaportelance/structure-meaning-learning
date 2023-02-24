import os, re, json
import numpy as np
import random
import torch
import torch.utils.data as data

TXT_IMG_DIVISOR=1
TXT_MAX_LENGTH=45

def set_constant(visual_mode, max_length):
    global TXT_IMG_DIVISOR, TXT_MAX_LENGTH
    TXT_IMG_DIVISOR = 1 if not visual_mode else 5
    TXT_MAX_LENGTH = max_length
    #print(TXT_IMG_DIVISOR, TXT_MAX_LENGTH)

def set_rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w

class EvalDataLoader(data.Dataset):
    def __init__(self, data_path, data_split, vocab, min_length=1, max_length=100):
        self.vocab = vocab
        self.captions = list()
        self.labels = list()
        self.spans = list()
        self.tags = list()
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f:
            for line in f:
                (caption, span, label, tag) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                if len(caption) < min_length or len(caption) > max_length:
                    continue
                self.captions.append(caption)
                self.labels.append(label)
                self.spans.append(span)
                self.tags.append(tag)
        self.length = len(self.captions)

    def __getitem__(self, index):
        caption = [self.vocab(token) for token in self.captions[index]]
        caption = torch.tensor(caption)
        label = self.labels[index]
        span = self.spans[index]
        tag = self.tags[index]
        return caption, label, span, tag, index

    def __len__(self):
        return self.length

def collate_fun_eval(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    zipped_data = list(zip(*data))
    captions, labels, spans, tags, ids = zipped_data
    max_len = max([len(caption) for caption in captions])
    targets = torch.zeros(len(captions), max_len).long()
    lengths = [len(cap) for cap in captions]
    for i, cap_len in enumerate(lengths):
        targets[i, : cap_len] = captions[i][: cap_len]
    return targets, lengths, spans, labels, tags, ids

def eval_data_iter(data_path, data_split, vocab, batch_size=128,
                   min_length=1, max_length=100):
    data = EvalDataLoader(data_path, data_split, vocab,
        min_length=min_length, max_length=max_length)
    data_loader = torch.utils.data.DataLoader(
                    dataset=data,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=None,
                    pin_memory=True,
                    collate_fn=collate_fun_eval
    )
    return data_loader

class SortedBlockSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        all_sample = len(self.data_source)
        batch_size = data_source.batch_size
        nblock = all_sample // batch_size
        residue = all_sample % batch_size
        nsample = all_sample - residue
        # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        self.groups = np.array_split(range(nsample), nblock)
        self.strip_last = False
        if residue > 0:
            self.strip_last = True
            block = np.array(range(nsample, all_sample))
            self.groups.append(block)

    def __iter__(self):
        self.data_source._shuffle()
        end = -1 if self.strip_last else len(self.groups)
        groups = self.groups[:end]
        #random.shuffle(groups)
        indice = torch.randperm(len(groups)).tolist()
        groups = [groups[k] for k in indice]
        if self.strip_last:
            groups.append(self.groups[-1])
        indice = list()
        for i, group in enumerate(groups):
            indice.extend(group)
            #print(i, group)
        assert len(indice) == len(self.data_source)
        return iter(indice)

    def __len__(self):
        return len(self.data_source)

class SortedRandomSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

class SortedSequentialSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class DataLoader(data.Dataset):
    def __init__(self, data_path, data_split, vocab,
                 load_img=True, img_dim=2048, batch_size=1, tiny=False):
        self.batch_size = batch_size
        self.vocab = vocab
        self.ids_captions_spans = list()
        max_length = TXT_MAX_LENGTH
        indexes, removed, idx = list(), list(), -1
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f1, open(os.path.join(data_path, f'{data_split}.id'), 'r') as f2:
            for line, img_id in zip(f1.readlines(), f2.readlines()):
                if tiny and idx > 32 :
                    break
                idx += 1
                (caption, span) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                if TXT_MAX_LENGTH < 1000 and (len(caption) < 2 or len(caption) > max_length):
                    removed.append((idx, len(caption)))
                    continue
                self.ids_captions_spans.append((int(img_id), caption, span))
                indexes.append(idx)
        self.length = len(self.ids_captions_spans)
        self.im_div = TXT_IMG_DIVISOR
        echo("removed idx: ")
        echo(removed)

        if load_img:
            self.images = np.load(os.path.join(data_path, 'all_resn-152.npy'))
        else:
            self.images = np.zeros(10020, img_dim)

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist()
        indice = sorted(indice, key=lambda k: len(self.ids_captions_spans[k]))
        self.ids_captions_spans = [self.ids_captions_spans[k] for k in indice]

    def __getitem__(self, index):
        img_id, cap, span = self.ids_captions_spans[index]
        image = torch.tensor(self.images[img_id])
        caption = [self.vocab(token) for token in cap]
        caption = torch.tensor(caption)
        span = torch.tensor(span)
        return image, caption, index, img_id, span

    def __len__(self):
        return self.length

def collate_fun(data):
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, ids, img_ids, spans = zipped_data
    images = torch.stack(images, 0)
    max_len = max([len(caption) for caption in captions])
    targets = torch.zeros(len(captions), max_len).long()
    lengths = [len(cap) for cap in captions]
    indices = torch.zeros(len(captions), max_len, 2).long()
    for i, cap in enumerate(captions):
        cap_len = len(cap)
        targets[i, : cap_len] = cap[: cap_len]
        indices[i, : cap_len - 1, :] = spans[i]
    return images, targets, lengths, ids, indices

def get_data_loader(data_path, data_split, vocab,
                    batch_size=128,
                    shuffle=True,
                    nworker=2,
                    loadimg=True,
                    img_dim=2048,
                    sampler=None,
                    tiny = False):
    dset = DataLoader(data_path, data_split, vocab, loadimg, img_dim, batch_size, tiny)
    if sampler:
        model = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            model = sampler
        #sampler = SortedRandomSampler(dset)
        sampler = model(dset)
    data_loader = torch.utils.data.DataLoader(
                    dataset=dset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True,
                    collate_fn=collate_fun
    )
    return data_loader

def get_data_iters(data_path, prefix, vocab, batch_size, nworker, shuffle=False, loadimg=True, sampler=True, tiny=False, split='train'):
    if split == 'val':
        sampler = None
    elif split == 'test':
        sampler = None
        shuffle=False
        loadimg=False
    data_loader = get_data_loader(data_path, prefix, vocab, batch_size=batch_size, shuffle=shuffle, nworker=nworker, sampler=sampler, loadimg=loadimg, tiny=tiny)
    return data_loader
