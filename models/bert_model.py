import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from pytorch_transformers import *


'''
BERT inputs are seq x batch
'''
class BModel(nn.Module):

    def __init__(self, eval_mode, embedding_type, model):
        super(BModel, self).__init__()
        if model == 'bert-full': 
            self.config = BertConfig(hidden_dropout_prob=0.01, attention_probs_dropout_prob=0.01, output_hidden_states=True)
            self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config) 
        elif model == 'gpt2': 
            self.model = GPT2Model.from_pretrained('gpt2')
        elif model == 'gpt2-large': 
            self.model = GPT2Model.from_pretrained('gpt2-large')

        self.eval_mode = eval_mode 
        self.embedding_type = embedding_type
  
    # input to this should be tokenized sentences using the BertTokenizer
    def forward(self, text):
        if self.eval_mode: 
            self.model.eval()
        
        last_hidden_states = []

        if self.embedding_type == 'sentence':
            for item in text: 
                final_rep = self.model(item.unsqueeze(0))[0].mean(dim=1)
                last_hidden_states.append(F.normalize(final_rep, dim=1))
        elif self.embedding_type == 'word':
            for item in text: 
                _, _, hiddens = self.model(item.unsqueeze(0))
                word_rep = hiddens[0].mean(dim=1)
                seq_rep = hiddens[-1].mean(dim=1)
                final_rep = torch.cat((word_rep, seq_rep), 1)
                last_hidden_states.append(F.normalize(final_rep, dim=1))
        else: 
            raise ValueError('embedding type not compatible with bert')
        
        return last_hidden_states









