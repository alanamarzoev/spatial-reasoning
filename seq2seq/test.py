import re
import spacy
import torch 
import numpy as np 
import sys 
import random
import pickle
from os import path
from torch.utils import data
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from sklearn.model_selection import train_test_split
from transformers import * 
from torch.nn import functional as F

dissimilar = 'move the yellow ball to the blue room'
similar = [('pick up grey ball', 'pick up the gray ball'), ('move to red room', 'go to the red room')]


def get_bert_embedding(sentence): 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])  
    out = []
    with torch.no_grad():
        _, _, hiddens = model(input_ids)
        word_rep = hiddens[0].mean(dim=1)
        seq_rep = hiddens[-1].mean(dim=1)
    out.append(F.normalize(word_rep, dim=1))
    out.append(F.normalize(seq_rep, dim=1))
    embedding = torch.cat(out, dim=1)
    return embedding 


def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2


def analyze_embeddings(): 
    for fake, real in similar: 
        fake_embedding = get_bert_embedding(fake)
        real_embedding = get_bert_embedding(real)
        dissimilar_embed = get_bert_embedding(dissimilar) 

        dist1 = dist_cos(fake_embedding, real_embedding)
        dist2 = dist_cos(fake_embedding, dissimilar_embed)
        dist3 = dist_cos(real_embedding, dissimilar_embed)

        mean = 0
        variance = 0.07 
        noise = np.random.normal(mean, variance/100, fake_embedding.shape)
        noised_embedding = np.add(fake_embedding, noise)
        # import ipdb; ipdb.set_trace()
        dist4 = dist_cos(fake_embedding, noised_embedding)
        print("COS DIST: {}".format(dist4))
        print("dist {} : {} = {}".format(fake, real, dist1))
        print("dist {} : {} = {}".format(fake, dissimilar, dist2))
        print("dist {} : {} = {}".format(real, dissimilar, dist3))
        # print('dist fake/real: {} dist fake/dissimilar: {} dist real/dissimilar: {}'.format(dist1, dist2, dist3))
        

if __name__=='__main__': 
    analyze_embeddings()