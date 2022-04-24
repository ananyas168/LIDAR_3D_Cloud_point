from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from Tnet import Tnet




class PointNet(nn.Module):
  def __init__(self, global_feat = True, Feature_Transform= True):
    super(PointNet,self).__init__()
    self.input_trans = Tnet(k=3)
    self.feature_trans = Tnet(k=64)
    self.conv1 = torch.nn.Conv1d(3,64,1)
    self.conv2 = torch.nn.Conv1d(64,128,1)
    self.conv3 = torch.nn.Conv1d(128,1024,1)
    self.fc1 = nn.Linear(1024,512)
    self.fc2 = nn.Linear(512,256)
    #self.fc3 = nn.Linear(256,k*k)
    #self.rl = nn.ReLU()

    self.bn1= nn.BatchNorm1d(64)
    self.bn2= nn.BatchNorm1d(128)
    self.bn3= nn.BatchNorm1d(1024)
    self.bn4= nn.BatchNorm1d(512)
    self.bn5= nn.BatchNorm1d(256)
    #self.bn= nn.BatchNorm1d(64)
    self.k = k
    self.global_feat=global_feat
    self.Feature_Transform = Feature_Transform

  def forward(self,x):
    batchsize= x.size()[0]
    no_of_points = x.size()[2]  
    #input transform:
    trans = self.input_trans(x) # shape input bx1x3, output: bx3x3
    x = x.transform(2,1) # shape bx3x1
    x = torch.bmm(trans,x).transform(2,1) #out bx1x3

    # applying conv1 to convert bx1x3 to bx1x64
    x = F.relu(self.bn1(self.conv1(x))) 
    if self.Feature_Transform:
      feat_trans = self.feature_trans(x) # shape input bx1x64, output: bx64x64
      
      x = x.transform(2,1) # shape bx64x1
      x = torch.bmm(trans,x).transform(2,1) #out bx1x64

    pointfeat = x
    x = F.relu(self.bn2(self.conv2(x))) # bx1x64 to bx1x128
    x = self.bn3(self.conv3(x)) # bx1x128 to bx1x1024
    x =  nn.MaxPool1d(x.size(-1))(x)
    x =  nn.Flatten(1)(x)           # bx1024

    if self.global_feat:
      return x

    else:
       x = x.view(-1, 1024, 1).repeat(1, 1, no_of_points)
       return torch.cat([pointfeat,x],1)
