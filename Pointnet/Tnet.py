from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F




class Tnet(nn.Module):
  def __init__(self,k):
    super(Tnet,self).__init__()
    self.conv1 = torch.nn.Conv1d(3,64,1)
    self.conv1 = torch.nn.Conv1d(64,128,1)
    self.conv1 = torch.nn.Conv1d(128,1024,1)
    self.fc1 = nn.Linear(1024,512)
    self.fc2 = nn.Linear(512,256)
    self.fc3 = nn.Linear(256,k*k)
    #self.rl = nn.ReLU()

    self.bn1= nn.BatchNorm1d(64)
    self.bn2= nn.BatchNorm1d(128)
    self.bn3= nn.BatchNorm1d(1024)
    self.bn4= nn.BatchNorm1d(512)
    self.bn5= nn.BatchNorm1d(256)
    #self.bn= nn.BatchNorm1d(64)
    self.k = k

  def forward(self,x):
    batchsize= x.size()[0]
    x = F.relu(self.bn1(self.conv1(x))) 
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x =  nn.MaxPool1d(x.size(-1))(x)
    x =  nn.Flatten(1)(x)
    x=  F.relu(self.bn4(self.fc1(x)))
    x=  F.relu(self.bn5(self.fc2(x)))
    x=  self.fc3(x) # weights 259x9  output: batch x 9

    #bias 1x9 np.eye(self.k).flatten().astype(np.float32)
    #identity = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,self.k*self.k).repeat(100,1)
    identity = Variable(torch.from_numpy(p.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(100,1)

    if x.is_cuda:
      identity.cuda()

    x = x+identity
    x= x.view(-1,self.k,self.k)
    return x  