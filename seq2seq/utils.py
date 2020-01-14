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


def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2


def save_synthetic_embeddings(sentences): 
    embeddings_path = 'embedded_synthetic.pkl'

    if not path.exists(embeddings_path): 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        all_info = {}

        for sentence in sentences: 
            input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])  
            out = []
            with torch.no_grad():
                _, _, hiddens = model(input_ids)
                word_rep = hiddens[0].mean(dim=1)
                seq_rep = hiddens[-1].mean(dim=1)
            out.append(F.normalize(word_rep, dim=1))
            out.append(F.normalize(seq_rep, dim=1))
            embedding = torch.cat(out, dim=1)
            all_info[sentence] = embedding 

        with open('embedded_synthetic.pkl', 'wb') as f: 
            pickle.dump(all_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return all_info 
    else: 
        with open(embeddings_path, 'rb') as handle:
            embeddings_info = pickle.load(handle)
            return embeddings_info 
            

def load_dataset(sents, word_to_idx):
    all_info = save_synthetic_embeddings(sents) 

    examples = []
    dataset = {}

    sum_dist = 0
    num_added = 0

    for sentence, embedding in all_info.items():
        x = sentence.split(" ")
        # import ipdb; ipdb.set_trace()
        # print(x)
        input_ids = torch.from_numpy(np.array([word_to_idx[word] for word in x]))
        examples.append((embedding, input_ids))
        for sentence2, embedding2 in all_info.items(): 
            if sentence != sentence2: 
                sum_dist += dist_cos(embedding, embedding2)
                num_added += 1 

    avg_dist = sum_dist / num_added

    # for sentence, embedding in all_info.items():
    #     embedding = embedding.numpy()
    #     mean = 0
        
    #     variance = avg_dist / len(embedding)
    #     print("VARIANCE: {}".format(variance))
    #     dataset[sentence] = {}

    #     input_ids = torch.from_numpy(np.array([word_to_idx[word] for word in x]))
    #     dataset[sentence]['tokenized'] = input_ids 
    dataset_size = len(examples)
    train = int(0.8 * dataset_size)
    valid = int(0.2 * dataset_size + train)
    
    return examples[0:train], examples[train:], avg_dist
