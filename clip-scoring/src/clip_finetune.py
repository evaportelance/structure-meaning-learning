import argparse
from tqdm import tqdm
import json
import pickle
from pathlib import Path
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

import os
import spacy, benepar
import torch.utils.data
from PIL import Image
import random
from utils.data import AbsScenesDataLoader, create_abstractscenes_img_list, create_abstractscenes_caps_dict, create_data_split


g_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
g_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../preprocessed-data/abstractscenes',
                        help='file path for preprocessed data')
    parser.add_argument('--result_dir', default='../results/abstractscenes',
                        help='directory where model, log files and eval results will be saved')
    parser.add_argument('--experiment_name', default='test', help='name of the experiment directory where model, log files and eval results will be stored')
    parser.add_argument('--as_img_dir', default='../../../AbstractScenes_v1.1/RenderedScenes', type=str, help='directory with AbstractScenes images')
    parser.add_argument('--prop', default=0.7, type=float, help='proportion of data to use as train data versus test data')
    parser.add_argument('--preprocessed_data', action='store_true', help='use existing preprocessed data')
    parser.add_argument('--preprocessing_num_workers', default=1, help='number of persistent workers for data preprocessing')
    parser.add_argument('--parse_diff', action='store_true', help='use parse differences for eval')
    parser.add_argument('--tiny', action='store_true', help='create tiny datasets for testing')
    parser.add_argument('--with_struct_loss', action='store_true', help='use constituent contrastive loss')
    parser.add_argument('--max_length', type=int, default=50, help='max caption length')
    parser.add_argument('--lr', default='5e-5', type=float,
                        help='learning rate for AdamW optimizer')
    parser.add_argument('--decay', default='0.1', type=float,
                        help='weight decay for AdamW optimizer')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


class AbsScenesDataset(torch.utils.data.Dataset):
    def __init__(self, ids, images, captions, trees, max_length):
        self.ids = ids
        self.images  = images
        self.captions = captions
        self.trees = trees
        self.max_length = max_length
        self.length = len(self.ids)


    def __getitem__(self, index):
        id = self.ids[index]
        caption = self.captions[index]
        image = self.images[index]
        tree = self.trees[index]
        return id, image, caption, tree, self.max_length


    def __len__(self):
        return self.length

# #    def _shuffle(self):
#         indice = torch.randperm(self.length).tolist()
#             indice = sorted(indice, key=lambda k: len(self.ids[k]))
#             self.ids_captions_spans = [self.ids_captions_spans[k] for k in indice]
# #
def collate_nostruct_fn(data):
    zipped_data = list(zip(*data))
    ids, images, captions, trees, max_length = zipped_data
    images = torch.stack(images)
    captions = g_processor.tokenizer(captions, max_length=max_length[0], padding="max_length", return_tensors="pt", truncation=True)

    batch = {
        "pixel_values": images,
        "input_ids": captions["input_ids"],
        "attention_mask": captions["attention_mask"],
    }

    return batch

def collate_struct_fn(data):
    zipped_data = list(zip(*data))
    ids, images, captions, trees, max_length = zipped_data
    tree_inputs = []
    tree_attention = []
    tree_lens = []
    tree_imgs = []
    for tree, img in zip(trees, images):
        tree_imgs += [img]*len(tree)
        constituents = g_processor.tokenizer(tree, max_length=max_length[0], padding="max_length", return_tensors="pt", truncation=True)
        tree_inputs += constituents["input_ids"]
        tree_attention += constituents["attention_mask"]
    images =  torch.stack(tree_imgs)
    input_ids = torch.stack(tree_inputs)
    attention_mask = torch.stack(tree_attention)

    batch = {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    return batch

def create_abstractscenes_datasets(args, data_path):
    random.seed(args.seed)
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})

    def get_data(data_dict, nlp):
        image_ids = data_dict.keys()
        ids = []
        images = []
        captions = []
        trees =[]
        print("parsing captions...")
        for im_id in tqdm(image_ids):
            i = 0
            img = data_dict[im_id]["img"]
            for cap in data_dict[im_id]["cap"]:
                ids.append((im_id*10+i))
                images.append(img)
                captions.append(cap)
                parse = nlp(cap)
                parse = list(parse.sents)[0]
                constituents = [str(x) for x in parse._.constituents]
                trees.append(constituents)
                i += 1
        return ids, images, captions, trees

    image_list = create_abstractscenes_img_list(args.as_img_dir)
    data_dict = create_abstractscenes_caps_dict(str(data_path))
    print("processing images...")
    for (id, img_file) in tqdm(image_list):
        img = Image.open(img_file).convert("RGB")
        img = g_processor.image_processor([img], return_tensors="pt")
        img = img['pixel_values']
        data_dict[id]["img"] = torch.reshape(img, (3, 224, 224))
    train_dict, test_dict = create_data_split(data_dict, args.prop)
    if args.tiny:
        train_dict = dict(list(train_dict.items())[:32])
        test_dict = dict(list(test_dict.items())[:6])
    ids, images, captions, trees = get_data(train_dict, nlp)
    train_dataset = AbsScenesDataset(ids, images, captions, trees, args.max_length)
    test_dataloader = AbsScenesDataLoader(test_dict, nlp, args.parse_diff)
    return train_dataset, test_dataloader



def train(args, model, optimizer, train_dataloader, device):
    model = model.train()
    print('Finetuning CLIP on abstractscenes datasets...')
    ### TRAINING ###
    num_batches = len(train_dataloader)
    for epoch in range(0, args.num_epochs):
        print("Epoch: " + str(epoch))
        total_loss = 0
        batch_n = 0
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch_n +=1
            optimizer.zero_grad()
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=True)
            loss = output.loss
            total_loss += loss
            loss.backward()
            optimizer.step()

            #if batch_n % 50 == 0:
                #avg_batch_loss = total_loss / batch_n
                #print("Average loss for "+str(batch_n)+"  batches:" + str(avg_batch_loss))
                #total_loss = 0
                #batch_n = 0
        avg_batch_loss = total_loss / batch_n
        print("Average loss for "+str(batch_n)+"  batches:" + str(avg_batch_loss))
    return model

### EVALUATION ###
def get_performance(scores):
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in scores:
      text_correct_count += 1 if text_correct(result) else 0
      image_correct_count += 1 if image_correct(result) else 0
      group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    text_score = text_correct_count/denominator
    image_score = image_correct_count/denominator
    group_score = group_correct_count/denominator

    return text_score, image_score, group_score

def get_similarity_score(text, image, model):
    input = g_processor(text=[text], images=[image], return_tensors="pt")
    pixel_values = input['pixel_values'].to(model.device)
    input_ids = input['input_ids'].to(model.device)
    attention_mask = input['attention_mask'].to(model.device)
    with  torch.no_grad():
        output = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        score = output.logits_per_image.item()
    return score


def get_constituent_score(id, images, captions, model):
  # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
  # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
    clip_score_c0_i0 = get_similarity_score(captions[0], images[0], model)
    clip_score_c1_i0 = get_similarity_score(captions[1], images[0], model)
    clip_score_c0_i1 = get_similarity_score(captions[0], images[1], model)
    clip_score_c1_i1 = get_similarity_score(captions[1], images[1], model)
    return {"id" : id, "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}

def get_multiconstituent_score(id, images, trees, model, parse_diff=False):
    constituents0, constituents1 = trees
    if parse_diff:
        constituents0_diff = list(set(constituents0).difference(set(constituents1)))
        constituents1_diff = list(set(constituents1).difference(set(constituents0)))
        constituents0 = constituents0_diff
        constituents1 = constituents1_diff
    norm0 = len(constituents0)
    norm1 = len(constituents1)
    constituents0_i0_scores = []
    constituents0_i1_scores = []
    constituents1_i0_scores = []
    constituents1_i1_scores = []
    const1_scores = []
    for c0 in constituents0:
        score_c0_i0 = get_similarity_score(c0, images[0], model)
        score_c0_i1 = get_similarity_score(c0, images[1], model)
        constituents0_i0_scores.append(score_c0_i0)
        constituents0_i1_scores.append(score_c0_i1)
    for c1 in constituents1:
        score_c1_i0 = get_similarity_score(c1, images[0], model)
        score_c1_i1 = get_similarity_score(c1, images[1], model)
        constituents1_i0_scores.append(score_c1_i0)
        constituents1_i1_scores.append(score_c1_i1)
    clip_score_c0_i0 = np.sum(constituents0_i0_scores) / norm0
    clip_score_c1_i0 = np.sum(constituents1_i0_scores) / norm1
    clip_score_c0_i1 =np.sum(constituents0_i1_scores) / norm0
    clip_score_c1_i1 = np.sum(constituents1_i1_scores) /norm1
    return {"id" : id, "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}

def get_scores(args, dataloader, model):
    nostruct_scores = list()
    struct_scores = list()
    print('Number of examples: '+str(len(dataloader)))
    for i in tqdm(range(len(dataloader))):
        example = dataloader[i]
        id = example['id']
        images = example['images']
        captions = example['captions']
        trees = example['trees']
        nostruct_score = get_constituent_score(id, images, captions, model)
        struct_score = get_multiconstituent_score(id, images, trees, model, parse_diff = args.parse_diff)

        nostruct_scores.append(nostruct_score)
        struct_scores.append(struct_score)

    return nostruct_scores, struct_scores

### MAIN ###
def main():
    args = get_args()
    experiment_dir = Path(args.result_dir) / args.experiment_name
    os.makedirs(str(experiment_dir), exist_ok=True)
    with open(str(experiment_dir / 'hyperparameters.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)
    torch.manual_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model = g_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.decay)
    ### DATA LOADER CREATION ###

    data_path = Path(args.data_dir)
    if args.preprocessed_data :
        print('Loading abstractscenes datasets with parses...')
        with open(str(data_path / 'as_test_data.pkl'), 'rb') as f:
            test_dataloader = torch.load(f, map_location=torch.device('cpu'))
        with open(str(data_path / 'as_train_data.pkl'), 'rb') as f:
            train_dataset = torch.load(f, map_location=torch.device('cpu'))
    else:
        print('Creating abstractscenes datasets with parses...')
        train_dataset, test_dataloader = create_abstractscenes_datasets(args, data_path)
        with open(str(data_path / 'as_train_data.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(str(data_path / 'as_test_data.pkl'), 'wb') as f:
            pickle.dump(test_dataloader, f)
    if args.with_struct_loss:
        collate_fn = collate_struct_fn
    else:
        collate_fn = collate_nostruct_fn
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    model = train(args, model, optimizer, train_dataloader, device)
    torch.save(model, str(experiment_dir / "model.pt"))

    print('Getting abstractscenes scores with and without parses...')
    as_nostruct_scores, as_struct_scores = get_scores(args, test_dataloader, model)
    with open(str(experiment_dir / 'as_nostruct_scores.json'), 'w') as f:
        json.dump(as_nostruct_scores, f)
    with open(str(experiment_dir / 'as_struct_scores.json'), 'w') as f:
        json.dump(as_struct_scores, f)

    print('Getting abstractscenes performance and writting results...')
    nostruct_text_score, nostruct_image_score, nostruct_group_score = get_performance(as_nostruct_scores)
    struct_text_score, struct_image_score, struct_group_score = get_performance(as_struct_scores)
    with open(str(experiment_dir / 'scores.csv'), 'w') as f:
        f.write('type, text score, image score, group score\n')
        f.write('no structure,'+str(nostruct_text_score) +', '+ str(nostruct_image_score) +', '+ str(nostruct_group_score)+'\n')
        f.write('with structure,'+str(struct_text_score) +', '+ str(struct_image_score) +', '+ str(struct_group_score)+'\n')


if __name__=="__main__":
    main()
