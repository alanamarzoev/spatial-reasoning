import torch 
import torchtext 
import time 
import random
import os 
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np 
from transformers import *
from torch.nn import functional as F
from utils import load_dataset
from torch import nn
from torch import optim
import pickle

SOS_token = 0
EOS_token = 1 
teacher_forcing_ratio = 0.5

def interactive_testing(decoder, max_length, idx_to_word): 
    sentence = str(input('test input:')) 
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
    decoder_hidden = embedding.unsqueeze(0)
    decoder_input = torch.tensor([[SOS_token]])
    output = []
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        output.append(decoder_input)
    en_sentence = " ".join(IDX_TO_WORD[x.item()] for x in output)
    print('output: {}'.format(en_sentence))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # import ipdb; ipdb.set_trace()
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def evaluate(iter,
             lowest, 
             validation, 
             decoder, 
             decoder_optimizer, 
             criterion, 
             max_length): 

    decoder.eval() 
    if lowest is None: 
        lowest = 20

    overall_loss = 0 
    for input_tensor, target_tensor in validation: 
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        encoder_hidden = input_tensor 
        
        loss = 0

        decoder_input = torch.tensor([[SOS_token]])

        decoder_hidden = encoder_hidden.unsqueeze(0).float()

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                # import ipdb; ipdb.set_trace()
                loss += criterion(decoder_output, target_tensor[[di]])

                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, target_tensor[[di]])
                if decoder_input.item() == EOS_token:
                    break
        # print("intermediate val loss: {}".format(loss.item()/target_length))
        overall_loss += loss.item() / target_length

    overall_loss = overall_loss / len(validation)
    if overall_loss < lowest: 
        if not os.path.isdir(".save"):
            os.makedirs(".save")
        torch.save(decoder.state_dict(), './.save/seq2seq_%d.pt' % (iter))

    return overall_loss
    

def train(input_tensor, 
        target_tensor, 
        decoder, 
        decoder_optimizer, 
        criterion, 
        max_length):

    # print('input_tensor: {}, target_tensor: {}'.format(input_tensor, target_tensor))

    decoder.train() 
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_hidden = input_tensor 
    
    loss = 0

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden.unsqueeze(0).float()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = False 

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # import ipdb; ipdb.set_trace()
            # print('step: {} decoder output: {} target output: {}'.format(di, decoder_output, target_tensor[[di]]))
            loss += criterion(decoder_output, target_tensor[[di]])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # print('step: {} decoder output: {} target output: {}'.format(di, decoder_output, target_tensor[[di]]))
            # print('topv: {} topi: {}'.format(topv, topi))
            loss += criterion(decoder_output, target_tensor[[di]])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(decoder, n_iters, dataset, word_to_idx, max_length, print_every=10, plot_every=100, learning_rate=0.001, test_every=200):
    print('loading dataset...')
    training, val, avg_dist = load_dataset(dataset, word_to_idx)
    random.shuffle(training)
    random.shuffle(val)

    start = time.time()
    plot_losses = []
    val_plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    training_pairs = training
    criterion = nn.NLLLoss()
    val_every = 1000
    lowest = None 
    mean = 0 

    for iter in range(1, n_iters + 1):
        # print('example #: {}'.format(y))
        y = random.randrange(len(training_pairs))
        training_pair = training_pairs[y]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        input_tens_np = input_tensor.numpy()
        # variance = avg_dist / (len(input_tens_np) * 2) 
        # print('variance: {}'.format(variance))
        
        noise = np.random.normal(0.05/len(input_tens_np), 0.04/len(input_tens_np), input_tens_np.shape)
        import ipdb; ipdb.set_trace()
        print("noise: {}".format(noise))
        noised_embedding = np.add(input_tens_np, noise)
        input_tensor = torch.from_numpy(noised_embedding)
        
        loss = train(input_tensor, target_tensor,
                     decoder, decoder_optimizer, criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0: 
            print('loss: {}'.format(loss))
            plot_losses.append(loss) 
        
        if iter % val_every == 0: 
            loss = evaluate(iter, lowest, val, decoder, decoder_optimizer, criterion, max_length)

            if lowest is None or loss < lowest: 
                lowest = loss 

            print('val loss: {}'.format(loss))
            print("val loss history: {}".format(val_plot_losses))
            val_plot_losses.append(loss)
        
    interactive_testing(decoder, max_length, idx_to_word)
    show_plot(plot_losses)


def main(): 
    sent_to_ind = pickle.load( open( "sent_to_ind.pkl", "rb" ) )
    sents = []
    words = set()
    max_len = 0
    for sent, ind in sent_to_ind.items(): 
        # import ipdb; ipdb.set_trace()
        sent = ' '.join(map(lambda x: x.strip('u').strip("'"), sent.strip('[').strip(']').split(', ')))  
        x = sent.split(" ")
        for x in x: 
            words.add(x)
        if len(x) > max_len: 
            max_len = len(x)

        # sent = " ".join(x)
        sents.append(sent)

    words = list(words)
    words.insert(0, "EOS")
    words.insert(0, "SOS")

    word_to_idx = {k:v for v, k in enumerate(words)}
    # import ipdb; ipdb.set_trace()
    output_size = len(words)
    hidden_size = 1536 # bert embedding size 
    decoder = DecoderRNN(hidden_size, output_size)
    train_iters(decoder, 500, sents, word_to_idx, max_len)


if __name__ == '__main__':
    main() 