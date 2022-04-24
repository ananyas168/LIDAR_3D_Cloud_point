from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from Pointnet import PointNet



class PointNetCls(nn.Module):
  def __init__(self, k = 3,Feature_Transform = True ):
    super(PointNetSeg,self).__init__()
    self.pointnet = PointNet(global_feat = True, Feature_Transform= True) #bx1024
    self.fc1 = nn.Linear(1024,512)
    self.fc2 = nn.Linear(512,256)
    self.fc3 = nn.Linear(256,k)
    #self.rl = nn.ReLU()
    self.bn1= nn.BatchNorm1d(512)
    self.bn2= nn.BatchNorm1d(256)
    self.dropout = nn.Dropout(p=0.3)
    self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
      x = self.pointnet(x)
      x=  F.relu(self.bn1(self.fc1(x)))
      x=  F.relu(self.bn2(self.dropout(self.fc2(x))))
      x=  self.fc3(x)
      return self.logsoftmax(x)