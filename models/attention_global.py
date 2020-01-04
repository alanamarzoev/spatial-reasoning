## attention model with only a single convolution
## LSTM kernel isn't really used as attention map,
## but just as a kernel for convolution

import math, pdb
import torch, torch.nn as nn, torch.nn.functional as F
import custom



def cudit(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


class AttentionGlobal(nn.Module):
# args.lstm_out, args.goal_hid, args.rank, args.obj_embed
    def __init__(self, text_model, args, map_dim = 10):
        super(AttentionGlobal, self).__init__()

        assert args.attention_kernel % 2 == 1
        self.args = args 
        self.text_model = text_model
        # self.object_model = object_model
        self.embed_type = args.embedding_type 
        self.embed_dim = args.attention_in_dim
        self.kernel_out_dim = args.attention_out_dim
        self.kernel_size = args.attention_kernel
        self.global_coeffs = args.global_coeffs
        self.model = args.model 
        self.reshape = cudit(torch.nn.Linear(768, 66))
        self.reshape_word = cudit(torch.nn.Linear(1536, 66))

        
    
        self.reshape_word_mlp = cudit(torch.nn.Linear(1536, 256))
        self.reshape_word_mlp2 = cudit(torch.nn.Linear(256, 66))

        self.reshape_mlp = cudit(torch.nn.Linear(768, 256))
        self.reshape_mlp2 = cudit(torch.nn.Linear(256, 66))

        padding = int(math.ceil(self.kernel_size/2.)) - 1

        self.conv_custom = custom.ConvKernel(self.embed_dim, self.kernel_out_dim, self.kernel_size, bias=False, padding=padding)
        self.reshape_dim = self.kernel_out_dim * (map_dim-self.kernel_size+1)**2

    def __conv(self, inp, kernel):
        batch_size = inp.size(0)
        out = [ self.conv_custom(inp[i].unsqueeze(0), kernel[i]) for i in range(batch_size) ]
        out = torch.cat(out, 0)
        return out

   
    def forward(self, inp):
        (embeddings, text) = inp
        if self.embed_type == 'lstm':
            batch_size  = embeddings.size(0)
            hidden = self.text_model.init_hidden(batch_size)
            text = text.transpose(0,1)
            lstm_out = self.text_model.forward(text, hidden)
            lstm_kernel = lstm_out[:,:-self.global_coeffs].contiguous()
            lstm_kernel = lstm_kernel.view(-1, self.kernel_out_dim, self.embed_dim, self.kernel_size, self.kernel_size)
            local_heatmap = self.__conv(embeddings, lstm_kernel)
            local_heatmap = local_heatmap.sum(1, keepdim=True)

            lstm_global = lstm_out[:,-self.global_coeffs:]
            self.output_local = local_heatmap
            self.output_global = lstm_global

            return local_heatmap, lstm_global

        elif self.embed_type == 'bert' or self.embed_type == 'bert-fixed' or self.embed_type == 'bert-word' or self.embed_type == 'bert-word-fixed': 
            batch_size = embeddings.size(0)
            bert_out = self.text_model(text)
        
            text_embed = []
            for embed in bert_out:
                if self.args.bert_reshape_type == 'linear':
                    if self.embed_type == 'bert-word' or self.embed_type == 'bert-word-fixed': 
                        lstm_out = self.reshape_word(embed)
                    else: 
                        lstm_out = self.reshape(embed)
                else: # MLP
                    if self.embed_type == 'bert-word' or self.embed_type == 'bert-word-fixed': 
                        lstm_out = self.reshape_word_mlp2(F.relu(self.reshape_word_mlp(embed)))
                    else: 
                        lstm_out = self.reshape_mlp2(F.relu(self.reshape_mlp(embed)))

                text_embed.append(lstm_out)
           
            lstm_out = torch.stack(text_embed).squeeze(1)
            lstm_kernel = lstm_out[:,:-self.global_coeffs].contiguous()
            lstm_kernel = lstm_kernel.view(-1, self.kernel_out_dim, self.embed_dim, self.kernel_size, self.kernel_size)
            local_heatmap = self.__conv(embeddings, lstm_kernel)
            local_heatmap = local_heatmap.sum(1, keepdim=True)
            lstm_global = lstm_out[:,-self.global_coeffs:]
            self.output_local = local_heatmap
            self.output_global = lstm_global

            return local_heatmap, lstm_global
        
        else: # assuming one-hot or random 
            batch_size  = embeddings.size(0)
            out = []
            if self.embed_type == 'one-hot': # not actually one hot, just zeros vec 
                out.append(torch.zeros(66)) 
                out = torch.stack(out)
            elif self.embed_type == 'random':
                out = [torch.rand(1, 66) for x in text]
                out = torch.stack(out)
                # out = text
                out = out.squeeze(1)
            else: 
                raise ValueError('embedding not recognized...')         

            lstm_kernel = out[:,:-self.global_coeffs].contiguous()
            lstm_kernel = lstm_kernel.view(-1, self.kernel_out_dim, self.embed_dim, self.kernel_size, self.kernel_size)
            local_heatmap = self.__conv(embeddings, lstm_kernel)
            local_heatmap = local_heatmap.sum(1, keepdim=True)
            lstm_global = out[:,-self.global_coeffs:]
            self.output_local = local_heatmap
            self.output_global = lstm_global

            return local_heatmap, lstm_global



if __name__ == '__main__':
    import argparse
    from text_model import *
    from object_model import *
    from lookup_model import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_kernel', type=int, default=3)
    parser.add_argument('--attention_out_dim', type=int, default=3)
    parser.add_argument('--obj_embed', type=int, default=5)
    parser.add_argument('--map_dim', type=int, default=10)
    args = parser.parse_args()

    batch = 2
    seq = 10

    text_vocab = 10
    lstm_inp = 5
    lstm_hid = 3
    lstm_layers = 1

    # obj_inp = torch.LongTensor(1,10,10).zero_()
    obj_vocab = 3
    # emb_dim = 3

    lstm_out = args.obj_embed * args.attention_out_dim * args.attention_kernel**2
    # hidden_dim = 5
    # out_dim = 7

    text_model = TextModel(text_vocab, lstm_inp, lstm_hid, lstm_layers, lstm_out)
    object_model = LookupModel(obj_vocab, args.obj_embed)

    psi = AttentionHeatmap(text_model, object_model, args, map_dim = args.map_dim)

    hidden = text_model.init_hidden(batch)
    text_inp = Variable(torch.floor(torch.rand(batch,seq)*text_vocab).long())
    obj_inp = Variable(torch.floor(torch.rand(batch,1,10,10)*obj_vocab).long())

    to = psi.forward( (obj_inp, text_inp) )
    print to.size()

    # # inp = Variable(torch.LongTensor((1,2,3)))
    # print 'INPUT: ', text_inp
    # print text_inp.size()
    # out = text_model.forward( text_inp, hidden )

    # print 'OUT: ', out
    # print out.size()
    # # print 'HID: ', hid

    # obj_inp = Variable(torch.LongTensor(5,1,20,20).zero_())
    # obj_inp = Variable(obj_inp)
    # obj_out = object_model.forward(obj_inp)

    # print obj_out.data.size()








