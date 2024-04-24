import os, re, json
import numpy as np
import random
import torch
import torch.utils.data as data
from operator import itemgetter

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

class AsDataset(data.Dataset):
    def __init__(self, data_path, data_split, vocab,
                 load_img=True, encoder_file='all_as-resn-50.npy', img_dim=2048, batch_size=1, tiny=False, one_shot=False, sem_data=False):
        self.batch_size = batch_size
        self.vocab = vocab
        self.ids_captions_spans = list()
        self.test_ids_contrastive = {'transitive':[], 'intransitive':[]}
        max_length = TXT_MAX_LENGTH
        removed, idx = list(), -1
        with open(os.path.join(data_path, 'test_verb_ids.json'), 'r') as f:
            test_ids = json.load(f)
        with open(os.path.join(data_path, 'test_verb_ids_indist.json'), 'r') as f:
            test_ids_indist = json.load(f)
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f1, open(os.path.join(data_path, f'{data_split}.id'), 'r') as f2:
            for line, img_id in zip(f1.readlines(), f2.readlines()):
                if tiny and idx > 1000 :
                    break
                idx += 1
                (caption, span) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                if TXT_MAX_LENGTH < 1000 and (len(caption) < 2 or len(caption) > max_length):
                    removed.append((idx, len(caption)))
                    continue
                if not sem_data:
                    if str(idx) in test_ids:
                        if one_shot:
                            v_type = test_ids[str(idx)]['v_type']
                            self.test_ids_contrastive[v_type].append((int(img_id), caption, span, idx))
                        else:
                            if idx in test_ids_indist:
                                v_type = test_ids[str(idx)]['v_type']
                                self.test_ids_contrastive[v_type].append((int(img_id), caption, span, idx))
                            else:
                                self.ids_captions_spans.append((int(img_id), caption, span, idx))  
                    else:
                        self.ids_captions_spans.append((int(img_id), caption, span, idx))
                else:
                    self.ids_captions_spans.append((int(img_id), caption, span, idx))
        self.length = len(self.ids_captions_spans)
        self.im_div = TXT_IMG_DIVISOR
        #print("removed idx: ")
        #print(removed)
        if load_img:
            self.images = np.load(os.path.join(data_path, encoder_file))
        else:
            self.images = np.zeros((10020, img_dim))

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist()
        #indice = sorted(indice, key=lambda k: len(self.ids_captions_spans[k]))
        self.ids_captions_spans = [self.ids_captions_spans[k] for k in indice]

    def __getitem__(self, index):
        img_id, cap, span, idx = self.ids_captions_spans[index]
        image = torch.tensor(self.images[img_id])
        caption = [self.vocab(token) for token in cap]
        caption = torch.tensor(caption)
        span = torch.tensor(span)
        return image, caption, idx, img_id, span

    def __len__(self):
        return self.length
    
class BiAsDataset(data.Dataset):
    def __init__(self, dset):
        self.batch_size = dset.batch_size
        self.vocab = dset.vocab
        self.test_ids_contrastive = dset.test_ids_contrastive
        self.images = dset.images
        self.length = min([len(self.test_ids_contrastive['transitive']),len(self.test_ids_contrastive['intransitive'])])

    def _shuffle(self):
        transitive_indice = torch.randperm(len(self.test_ids_contrastive['transitive'])).tolist()
        intransitive_indice = torch.randperm(len(self.test_ids_contrastive['intransitive'])).tolist()
        #indice = sorted(indice, key=lambda k: len(self.ids_captions_spans[k]))
        self.test_ids_contrastive['transitive'] = [self.test_ids_contrastive['transitive'][k] for k in transitive_indice]
        self.test_ids_contrastive['intransitive'] = [self.test_ids_contrastive['intransitive'][k] for k in intransitive_indice]

    def __getitem__(self, index):     
        img_id_transitive, cap_transitive, span_transitive, idx_transitive = self.test_ids_contrastive['transitive'][index]
        img_id_intransitive, cap_intransitive, span_intransitive, idx_intransitive = self.test_ids_contrastive['intransitive'][index]
        image_transitive = torch.tensor(self.images[img_id_transitive])
        image_intransitive = torch.tensor(self.images[img_id_intransitive])
        caption_transitive = [self.vocab(token) for token in cap_transitive]
        caption_transitive = torch.tensor(caption_transitive)
        caption_intransitive = [self.vocab(token) for token in cap_intransitive]
        caption_intransitive = torch.tensor(caption_intransitive)
        span_transitive = torch.tensor(span_transitive)
        span_intransitive = torch.tensor(span_intransitive)
        return image_transitive, image_intransitive, caption_transitive, caption_intransitive, idx_transitive, idx_intransitive, img_id_transitive, img_id_intransitive, span_transitive, span_intransitive 

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
    
def bi_collate_fun(data):
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    zipped_data = list(zip(*data))
    images_transitive, images_intransitive, captions_transitive, captions_intransitive, idx_transitive, idx_intransitive, img_ids_transitive, img_ids_intransitive, spans_transitive, spans_intransitive = zipped_data
    images_tr = torch.stack(images_transitive, 0)
    images_intr = torch.stack(images_intransitive, 0)
    max_len_tr = max([len(cap) for cap in captions_transitive])
    max_len_intr = max([len(cap) for cap in captions_intransitive])
    max_len = max(max_len_tr, max_len_intr)
    targets_tr = torch.zeros(len(captions_transitive), max_len).long()
    targets_intr = torch.zeros(len(captions_intransitive), max_len).long()
    lengths_tr = [len(cap) for cap in captions_transitive]
    lengths_intr = [len(cap) for cap in captions_intransitive]
    indices_tr = torch.zeros(len(captions_transitive), max_len, 2).long()
    indices_intr = torch.zeros(len(captions_intransitive), max_len, 2).long()
    for i, cap in enumerate(captions_transitive):
        cap_len = len(cap)
        targets_tr[i, : cap_len] = cap[: cap_len]
        indices_tr[i, : cap_len - 1, :] = spans_transitive[i]
    for i, cap in enumerate(captions_intransitive):
        cap_len = len(cap)
        targets_intr[i, : cap_len] = cap[: cap_len]
        indices_intr[i, : cap_len - 1, :] = spans_intransitive[i]
    # targets are captions, indices are gold spans
    return images_tr, targets_tr, lengths_tr, idx_transitive, indices_tr, images_intr, targets_intr, lengths_intr, idx_intransitive, indices_intr



def get_data_iters(data_path, data_split, vocab,
                    batch_size=128,
                    nworker=2,
                    shuffle=True,
                    sampler=True,
                    load_img=True,
                    encoder_file = 'all_as-resn-50.npy',
                    img_dim=2048,
                    tiny = False,
                    one_shot=True):
    dset = AsDataset(data_path, data_split, vocab, load_img, encoder_file, img_dim, batch_size, tiny, one_shot=one_shot)
    dset_all = AsDataset(data_path, data_split, vocab, load_img, encoder_file, img_dim, batch_size, tiny, one_shot=False, sem_data=True)
    if sampler:
        model = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            model = sampler
        #sampler = SortedRandomSampler(dset)
        sampler = model(dset)
    syn_test_dset = BiAsDataset(dset)
    train_data_loader = torch.utils.data.DataLoader(
                    dataset=dset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True,
                    collate_fn=collate_fun)
    syn_test_data_loader = torch.utils.data.DataLoader(
                    dataset=syn_test_dset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    pin_memory=True,
                    collate_fn=bi_collate_fun)
    sem_test_data_loader = torch.utils.data.DataLoader(
                    dataset=dset_all,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=None,
                    pin_memory=True,
                    collate_fn=collate_fun)
    return train_data_loader, syn_test_data_loader, sem_test_data_loader