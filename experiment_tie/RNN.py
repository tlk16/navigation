#  This file is to generate the network and train it 
#  Vanilla RNN with the inertia controls internal timescale   
#  random initialization
#  weights used xavier initialization
#  action feedback is by default zero 

import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import init
from torch.nn import DataParallel

import os
import psutil
import gc


class RNN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, output_size, inertia = 0.5, k_action = 1, max_size = 20):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Parameter(torch.randn(input_size, hidden_size) * 10 * np.sqrt(2.0/(input_size + hidden_size)))
        self.h2h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 1.0 * np.sqrt(2.0/hidden_size))
        self.h2o = nn.Parameter(torch.randn(hidden_size, output_size) * 1 * np.sqrt(2.0/(hidden_size + output_size)))
        self.a2h = nn.Parameter(torch.randn(action_size, hidden_size) * 1 * np.sqrt(2.0/(hidden_size + 4)))
        self.bh = nn.Parameter(torch.zeros(1, hidden_size))
        self.bo = nn.Parameter(torch.zeros(1, output_size))
        self.r = nn.Parameter(inertia * torch.ones(1, hidden_size))


    def forward(self, input_, hidden, action):
        # dim should be same except catting dimension
        # print(input_.shape, hidden.shape, action.shape)
        hidden_ = torch.tanh(input_.matmul(self.i2h) + hidden.matmul(self.h2h) + action.matmul(self.a2h) + self.bh)
        hidden = torch.mul((1 - self.r), hidden_) + torch.mul(self.r, hidden) 
        output = hidden.matmul(self.h2o)+ self.bo
        return output, hidden

    def forward_sequence_values(self, inputs, hidden0, actions):
        """

        :param inputs: [batch_stize, sequence_len, input_size]
        :param hidden0: [batch_size, input_size]
        :param actions: [batch_stize, sequence_len, action_size]
        :return:
        """
        # print('actions', actions)
        # print(inputs.shape)
        squence_length = inputs.shape[1]
        outputs = []
        hiddens = []
        hidden = torch.squeeze(hidden0)
        for i in range(squence_length):
            # print(inputs[:, i].shape, hidden.shape, actions[:, i].shape)
            output, hidden = self.forward(inputs[:, i], hidden, actions[:, i])
            outputs.append(output)
            hiddens.append(hidden)

        return outputs, hiddens


    

    def initHidden(self, batchsize = 1):
        return Variable(torch.randn(batchsize, self.hidden_size))
    
    @ staticmethod
    def crossentropy(predict, target, batch_size, beta = 1e-2):
        return torch.mean(-F.softmax(- beta * target.view(batch_size,-1)) \
                        * torch.log(F.softmax(- beta * predict.view(batch_size,-1), dim = 1) + 1e-5)) 
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.h2p = nn.Parameter(torch.randn(hidden_size, output_size) * 1 * np.sqrt(2.0/(hidden_size + 2 * output_size + 8)))
        self.bp = nn.Parameter(torch.randn(1, output_size) * 1 * np.sqrt(2.0/(hidden_size + 2 * output_size + 8)))

    def forward(self, hidden):
        # dim should be same except catting dimension
        output = hidden.matmul(self.h2p)+ self.bp
        return output

    def forward_sequence_values(self, hiddens):
        """

        :param hiddens: [batch_stize, sequence_len, hidden_size]
        :return:
        """
        # print('actions', actions)
        # print(inputs.shape)
        squence_length = hiddens.shape[1]
        outputs = []
        for i in range(squence_length):
            # print(inputs[:, i].shape, hidden.shape, actions[:, i].shape)
            output = self.forward(hiddens[:, i])
            outputs.append(output)
        return outputs
    
# recursive least square is used, mtraix P which is inverse correlaiton of input and beta is updated in a recursive way, pay attention to memory related to matrix inversion  
class LinearRegression(torch.nn.Module):
    def __init__(self, alpha):
        super(LinearRegression, self).__init__()
        self.alpha = alpha
    # choose an alpha to regulizer, alpha determines strength of regulazation  
    # leak not in inversion here 
    def LeastSquare(self, input_, target): 
        self.cov = input_.t().matmul(input_).data.numpy()
        P = torch.inverse(torch.from_numpy(self.cov).float() + self.alpha * torch.eye(input_.size()[1]))
        self.beta = P.matmul(input_.t()).matmul(target)
     
    
class RLS(LinearRegression):
    # get the first N data points and start, data x here is concat version of x and 1
    def __init__(self, alpha, lam = 0.5): 
        LinearRegression.__init__(self, alpha)
        # forget factor 
        self.lam = lam
    # update inverse correlation matrix by new data    
        
    def update_beta(self, x1, y1, trial):
        # kalman gain
        self.cov  = self.lam * self.cov + x1.t().matmul(x1).data.numpy() 
        P = np.linalg.inv(self.cov + self.alpha * np.eye(x1.size()[1]))
        Pr = P @ x1.data.numpy().T
        #  get error
        err = y1.data.numpy() - x1.data.numpy() @ self.beta.data.numpy()
        dbeta = Pr @ err
        self.beta += torch.from_numpy(dbeta).float()
#         err2 = y1.data.numpy() - x1.data.numpy() @ self.beta.data.numpy()
#         print ('triall', trial, 'err', np.sum(err), 'err2', np.sum(err2))
       
if __name__ == "__main__":
    rnn = RNN(input_size=4, action_size=2, hidden_size=512, output_size=8)
    action = torch.zeros((2,))
    input = torch.Tensor([1, 2, 3, 4])
    h0 = rnn.initHidden()
    print(input.dtype, h0.dtype, action.dtype)
    action, hidden = rnn(input, h0, action)
