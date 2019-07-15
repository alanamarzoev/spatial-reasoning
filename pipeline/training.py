import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

def cudit(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


class Trainer:
    def __init__(self, model, lr, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.criterion = cudit(nn.MSELoss(size_average=True))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def __epoch(self, inputs, targets, repeats = 1):
        self.model.train()
        if type(inputs) == tuple:
            data_size = inputs[0].size(0)
        else:
            data_size = inputs.size(0)
        num_batches = int(math.ceil(data_size / float(self.batch_size)) * repeats)

        err = 0
        for i in range(num_batches):
            inp, targ = self.__get_batch(inputs, targets)
            self.optimizer.zero_grad()
            out = self.model.forward(inp)
            loss = self.criterion(out, targ)
            loss.backward()
            self.optimizer.step()
            err += loss.data[0]
        err = err / float(num_batches)
        return err

    def __get_batch(self, inputs, targets):
        data_size = targets.size(0)

        inds = cudit(torch.floor(torch.rand(self.batch_size) * data_size).long())
        # bug: floor(rand()) sometimes gives 1
        inds[inds >= data_size] = data_size - 1

        if type(inputs) == tuple:
            inp = tuple([Variable( cudit(i.index_select(0, inds)) ) for i in inputs])
        else:
            inp = Variable( cudit(inputs.index_select(0, inds)) )

        targ = Variable( cudit(targets.index_select(0, inds)) )
        return inp, targ

    def train(self, inputs, targets, val_inputs, val_targets, iters = 10):
        t = trange(iters)
        for i in t:
            err = self.__epoch(inputs, targets)
            t.set_description( str(err) )
        return self.model
