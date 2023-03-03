import argparse
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from utils.data import get_winoground_data

import requests

# GLOBAL
g_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
g_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

def get_similarity_score(text, image):
    input = g_processor(text=[text], images=[image], return_tensors="pt")
    output = g_model(**input)
    score = output.logits_per_image.item()
    return score


def get_constituent_score(id, images, captions):
  # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
  # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
    clip_score_c0_i0 = get_similarity_score(captions[0], images[0])
    clip_score_c1_i0 = get_similarity_score(captions[1], images[0])
    clip_score_c0_i1 = get_similarity_score(captions[0], images[1])
    clip_score_c1_i1 = get_similarity_score(captions[1], images[1])
    return {"id" : id, "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}

def get_multiconstituent_score(id, images, trees):
    constituents0, constituents1 = trees
    norm0 = len(constituents0)
    norm1 = len(constituents1)
    constituents0_i0_scores = []
    constituents0_i1_scores = []
    constituents1_i0_scores = []
    constituents1_i1_scores = []
    const1_scores = []
    for c0 in constituents0:
        score_c0_i0 = get_similarity_score(c0, images[0])
        score_c0_i1 = get_similarity_score(c0, images[1])
        constituents0_i0_scores.append(score_c0_i0)
        constituents0_i1_scores.append(score_c0_i1)
    for c1 in constituents1:
        score_c1_i0 = get_similarity_score(c1, images[0])
        score_c1_i1 = get_similarity_score(c1, images[1])
        constituents1_i0_scores.append(score_c1_i0)
        constituents1_i1_scores.append(score_c1_i1)
    clip_score_c0_i0 = np.sum(constituents0_i0_scores) / norm0
    clip_score_c1_i0 = np.sum(constituents1_i0_scores) / norm1
    clip_score_c0_i1 =np.sum(constituents0_i1_scores) / norm0
    clip_score_c1_i1 = np.sum(constituents1_i1_scores) /norm1
    return {"id" : id, "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}

def get_winoground_scores(wino_dataloader):
    wino_nostruct_scores = list()
    wino_struct_scores = list()
    print('Number of examples: '+str(len(wino_dataloader)))
    for i in tqdm(range(len(wino_dataloader))):
        example = wino_dataloader[i]
        id = example['id']
        images = example['images']
        captions = example['captions']
        trees = example['trees']

        nostruct_score = get_constituent_score(id, images, captions)
        struct_scores = get_multiconstituent_score(id, images, trees)

    wino_nostruct_scores.append(nostruct_score)
    wino_struct_scores.append(struct_score)

    return wino_nostruct_scores, wino_struct_scores

def get_abstractscenes_scores(abstractscenes_dataloader):
    pass

if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()

    # Parser: Generative model parameters
    parser.add_argument('--wino_token', default='0', type=str, help='Hugging face authorization token to download Winoground')
    parser.add_argument('--odir', default='./', type=str, help='filename for results')
    parser.add_argument('--ofile', default='clip_test_results.csv', type=str, help='filename for results')
    parser.add_argument('--seed', default=30, type=int, help='random seed to use')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Creating winoground dataset with parses...')
    wino_dataloader = get_winoground_data(args)
    #abstractscenes_dataloader = get_abstractscenes_data(args)

    print('Getting winoground scores with and without parses...')
    wino_nostruct_scores, wino_struct_scores = get_winoground_scores(wino_dataloader)
    
    print('Getting winoground performance and writting results...')
    nostruct_text_score, nostruct_image_score, nostruct_group_score = get_performance(wino_nostruct_scores)
    struct_text_score, struct_image_score, struct_group_score = get_performance(wino_struct_scores)
    with open(args.ofile, 'w') as f:
        f.write('type, text score, image score, group score\n')
        f.write('no structure,'+str(nostruct_text_score) +', '+ str(nostruct_image_score) +', '+ str(nostruct_group_score)+'\n')
        f.write('with structure,'+str(struct_text_score) +', '+ str(struct_image_score) +', '+ str(struct_group_score)+'\n')


#inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
#outputs = model(**inputs)
#outputs.text_embeds
#outputs.image_embeds

# outputs.logits_per_image here size 1*2 and  outputs.logits_per_text here size 2:1
#logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
