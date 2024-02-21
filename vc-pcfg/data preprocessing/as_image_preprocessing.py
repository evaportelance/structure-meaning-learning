import os, sys, re
import copy, time, pickle, json
import numpy as np
import argparse
import csv

from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
from torchvision.transforms.functional import InterpolationMode
#from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision.models as models
#import torchvision.datasets as datasets
import torchvision.transforms as transforms
import resnet

# from clip import load
import random

def seed_all_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_all_rng(527)

# Data path options
parser = argparse.ArgumentParser()
#parser.add_argument('--data_root', default='', type=str, help='')
parser.add_argument('--npz_token', default='as-resn-50', type=str, help='')
#EP: Adding abstract scenes path arguments
parser.add_argument('--abstractscenes_root', default='../../../AbstractScenes_v1.1', type=str, help='')
parser.add_argument('--abstractscenes_out_root', default='../../preprocessed-data/abstractscenes', type=str, help='')
parser.add_argument('--checkpoint_path', default='../../pytorch-simclr/checkpoint/ckpt.pth', type=str, help='')
parser.add_argument('--split_list_file', default='', type=str, help='')
parser.add_argument('--split_name', default='', type=str, help='')
parser.add_argument('--clip_model_root', default='', type=str, help='')
parser.add_argument('--clip_model_name', default='', type=str, help='')
parser.add_argument('--batch_size', default=8, type=int, help='')
parser.add_argument('--peep_rate', default=100, type=int, help='')
parser.add_argument('--num_proc', default=1, type=int, help='')

cfg = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def build_clip_encoder(cfg):
#     model, _ = load(
#         cfg.clip_model_name, cfg.clip_model_root, device="cpu", jit=False
#     )
#     model = model.train(False)
#     return model

def build_as_resnet50_encoder(cfg):
    model = resnet.ResNet50(stem=resnet.StemImageNet)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model = model.train(False)
    return model

def build_resnet152_encoder(cfg):
    resnet101 = models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
    model = model.train(False)
    return model

def as_resnet50_transform(resize_size=224):
    mean = (0.428, 0.696, 0.526)
    std = (0.197, 0.179, 0.298)
    return transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])

def resnet_transform(resize_size=256, crop_size=224):
    """ (1) https://github.com/pytorch/vision/blob/d2bfd639e46e1c5dc3c177f889dc7750c8d137c7/references/classification/train.py#L111
        (2) https://github.com/pytorch/vision/blob/65676b4ba1a9fd4417293cb16f690d06a4b2fb4b/references/classification/presets.py#L44
        (3) or https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L105
        (4) or https://pytorch.org/hub/pytorch_vision_resnet
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        #transforms.PILToTensor(),                   # equal to ...
        #transforms.ConvertImageDtype(torch.float),  # ... ToTensor()
        transforms.Normalize(mean=mean, std=std),
    ])
    """
    PILToTensor() results in warnings: torchvision/transforms/functional.py:169: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:143.
    """

def clip_transform(n_px=224):
    """ https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/clip.py#L76
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

## EP: Abstract Scenes doesn't have train/val/test splits yet
def create_abstractscenes_data_list(cfg):
    """ all abstractscenes images. http://optimus.cc.gatech.edu/clipart/dataset/AbstractScenes_v1.1.zip
    """
    image_list = list()
    root = f"{cfg.abstractscenes_root}/RenderedScenes"
    for root, dir, files in os.walk(root):
        if len(dir) > 0:
            continue
        for fname in files:
            if fname.endswith(".png"):
                id = fname[:-4].replace("Scene", "").replace("_", "")
                image_list.append((id, f"{root}/{fname}"))
    return image_list


class ImageDatasetSrc(torch.utils.data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_list, transform=None):
        self.dataset = list()
        for i, line in enumerate(data_list):
            self.dataset.append(line)
        self.transform_resnet = resnet_transform()
        self.transform_clip = clip_transform()
        self.length = len(self.dataset)
        self.cfg = cfg

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        record = self.dataset[index]
        image = Image.open(record[1])
        image_resnet = self.transform_resnet(image)
        image_clip = self.transform_clip(image)
        item = {"name": record[0], "image_resnet": image_resnet, "image_clip": image_clip}
        return item

class ImageCollator:
    def __init__(self, device=torch.device("cpu")):
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # when pin_memory is true, the collator has to return CPU tensors
        self.device = device

    def __call__(self, records):
        union = {
            k: [record.get(k) for record in records] for k in set().union(*records)
        }
        return (
            torch.stack(union["image_clip"], dim=0),
            torch.stack(union["image_resnet"], dim=0),
            union["name"],
        )

def build_image_loader(cfg, data_list, transform):
    dataset = ImageDatasetSrc(cfg, data_list, transform)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=ImageCollator(),
        num_workers=cfg.num_proc,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    return dataloader

def save_image_npz(names, clip_npz_root, resnet_npz_root, z_clip=None, z_resnet=None):
    for i, name in enumerate(names):
        if z_clip is not None:
            np.savez_compressed(
                f"{clip_npz_root}/{name}", v=z_clip[i]
            )
        if z_resnet is not None:
            np.savez_compressed(
                f"{resnet_npz_root}/{name}", v=z_resnet[i]
            )

def encode_images(cfg, clip, resnet, dataloader, clip_npz_root, resnet_npz_root):
    nsample = 0
    start_time = time.time()
    for ibatch, (image_clip, image_resnet, names) in enumerate(dataloader):
        image_clip = image_clip.cuda(0, non_blocking=True)
        image_resnet = image_resnet.cuda(0, non_blocking=True)
        if clip is not None:
            z_clip = clip.encode_image(image_clip)
            z_clip = z_clip.cpu().numpy()
        else:
            z_clip = None
        z_resnet = resnet(image_resnet).squeeze()
        z_resnet = z_resnet.cpu().numpy()
        save_image_npz(names, clip_npz_root, resnet_npz_root, z_clip=z_clip, z_resnet=z_resnet)
        nsample += image_clip.shape[0]
        if (ibatch + 1) % cfg.peep_rate == 0:
            print(f"--step {ibatch + 1:08d} {nsample / (time.time() - start_time):.2f} samples/s")

def main_encode_abstractscenes_images(cfg):
    clip = None #build_clip_encoder(cfg).cuda()
    if cfg.npz_token == "as-resn-50":
        resnet = build_as_resnet50_encoder(cfg)
        transform = as_resnet50_transform()
        resnet_npz_root = f"{cfg.abstractscenes_root}/as-resn-50"
    else:
        resnet = build_resnet152_encoder(cfg).cuda()
        transform = resnet_transform()
        resnet_npz_root = f"{cfg.abstractscenes_root}/resn-152"

    abstractscenes_images = create_abstractscenes_data_list(cfg)
    abstractscenes_loader = build_image_loader(cfg, abstractscenes_images, transform)
    print(f"Total {len(abstractscenes_loader)} / {len(abstractscenes_loader.dataset)} AbstractScenes batches.")

    clip_npz_root = f"{cfg.abstractscenes_root}/clip-b32"

    for output_dir in [clip_npz_root, resnet_npz_root]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    encode_images(
        cfg, clip, resnet, abstractscenes_loader, clip_npz_root, resnet_npz_root
    )

def main_collect_abstractscenes_npz(cfg):
    def per_split(id_file, ipath, ofile):
        vectors = list()
        if id_file:
            with open(id_file, "r") as fr:
                for line in fr:
                    #assumes ids are in order
                    id = line.strip()
                    npz_file = f"{ipath}/{id}.npz"
                    vector = np.load(npz_file)["v"]
                    vectors.append(vector)
        else:
            for id in range(0, 10020):
                if id < 10:
                    npz_file = f"{ipath}/0{id}.npz"
                else:
                    npz_file = f"{ipath}/{id}.npz"
                vector = np.load(npz_file)["v"]
                vectors.append(vector)
        vectors = np.stack(vectors, axis=0)
        np.save(ofile, vectors)
        print(f"saved {vectors.shape} in {ofile}")

    if len(cfg.split_list_file) > 0:
        split_name = cfg.split_name
        npz_root = f"{cfg.abstractscenes_root}/{cfg.npz_token}"
        ofile = f"{cfg.abstractscenes_out_root}/{split_name}_{cfg.npz_token}.npy"
        id_file = f"{cfg.abstractscenes_out_root}/{cfg.split_list_file}"
        per_split(id_file, npz_root, ofile)
    else:
        split_name = "all"
        npz_root = f"{cfg.abstractscenes_root}/{cfg.npz_token}"
        ofile = f"{cfg.abstractscenes_out_root}/{split_name}_{cfg.npz_token}.npy"
        per_split(None, npz_root, ofile)


## Vectorize the labels from the original AS data file format
def get_gold_feature_classes(cfg):
    meta_info_file = f"{cfg.abstractscenes_root}/Scenes_10020.txt"
    png_class_file = f"{cfg.abstractscenes_root}/png_id_class.txt"
    png_id_class = dict()
    with open(png_class_file) as f:
        for line in csv.reader(f, delimiter="\t"):
            png_id_class[line[0]] = [int(line[1]),int(line[2])]
    img_features = []
    img_classes = []
    id = -1
    features = np.zeros((126, 6))
    classes = np.zeros(58)
    with open(meta_info_file) as f:
        for line in csv.reader(f, delimiter="\t"):
            if len(line) == 2:
                if id >= 0:
                    img_features.append(features)
                    img_classes.append(classes)
                id += 1
                features = np.zeros((126, 6))
                classes = np.zeros(58)
            elif len(line) == 7:
                item, _class = png_id_class[line[0]]
                features[item] = list(map(int, line[1:]))
                classes[_class] = 1
        img_features.append(features)
        img_classes.append(classes)
    return(img_features, img_classes)

## Create numpy array of the same length as text items (so including copies of images)
def main_collect_abstractscenes_labels(cfg):
    img_features, img_classes = get_gold_feature_classes(cfg)
    def per_split(id_file, feat_ofile, label_ofile):
        feat_vectors = []
        class_vectors = []
        if id_file:
            with open(id_file, "r") as fr:
                for line in fr:
                    #assumes ids are in order
                    id = int(line.strip())
                    feat_vectors.append(img_features[id])
                    class_vectors.append(img_classes[id])
        else:
            for id in range(0, 10020):
                feat_vectors.append(img_features[id])
                class_vectors.append(img_classes[id])
        feat_vectors = np.stack(feat_vectors, axis=0)
        class_vectors = np.stack(class_vectors, axis=0)
        np.save(feat_ofile, feat_vectors)
        np.save(label_ofile, class_vectors)
        
        print(f"saved {feat_vectors.shape} in {feat_ofile}")
        print(f"saved {class_vectors.shape} in {label_ofile}")

    if len(cfg.split_list_file) > 0:
        split_name = cfg.split_name
        feat_ofile = f"{cfg.abstractscenes_out_root}/{split_name}_features_gold.npy"
        label_ofile = f"{cfg.abstractscenes_out_root}/{split_name}_labels_gold.npy"
        id_file = f"{cfg.abstractscenes_out_root}/{cfg.split_list_file}"
        per_split(id_file, feat_ofile, label_ofile)
    else:
        feat_ofile = f"{cfg.abstractscenes_out_root}/all_features_gold.npy"
        label_ofile = f"{cfg.abstractscenes_out_root}/all_labels_gold.npy"
        #id_file = f"{cfg.abstractscenes_out_root}/all.id"
        id_file=None
        per_split(id_file, feat_ofile, label_ofile)
    class_ofile = f"{cfg.abstractscenes_out_root}/img_classes.npy"
    img_classes = np.stack(img_classes, axis=0)
    np.save(class_ofile, img_classes)
    print(f"saved {img_classes.shape} in {class_ofile}")
    
def main_flatten_labels(cfg):
    ifile = f"{cfg.abstractscenes_out_root}/all_features_gold.npy"
    gold_feats = np.load(ifile)
    assert gold_feats.shape == (10020, 126, 6)
    gold_feats = np.reshape(gold_feats, (10020, 756)).astype(np.float64)
    ofile = f"{cfg.abstractscenes_out_root}/all_flat_features_gold.npy"
    np.save(ofile, gold_feats)
    print(f"saved {gold_feats.shape} in {ofile}")
    
    

if __name__ == '__main__':
    print(cfg)
    with torch.no_grad():
        #main_encode_abstractscenes_images(cfg)
        pass
    #main_collect_abstractscenes_npz(cfg)
    #main_collect_abstractscenes_labels(cfg)
    main_flatten_labels(cfg)
    
    