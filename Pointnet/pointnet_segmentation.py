from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from Pointnet import PointNet


class PointNetSeg(nn.Module):
  def __init__(self, k = 3,Feature_Transform = True ):
    super(PointNetCls,self).__init__()
    self.k = k
    self.pointnet = PointNet(global_feat = False, Feature_Transform= True) #bx1024
    self.conv1 = torch.nn.Conv1d(1024,512,1)
    self.conv2 = torch.nn.Conv1d(512,256,1)
    self.conv3 = torch.nn.Conv1d(256,128,1)
    self.conv4 = torch.nn.Conv1d(128,self.k,1)
    #self.rl = nn.ReLU()
    self.bn1= nn.BatchNorm1d(512)
    self.bn2= nn.BatchNorm1d(256)
    self.bn3= nn.BatchNorm1d(128)
    self.dropout = nn.Dropout(p=0.3)
    self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
      batchsize= x.size()[0]
      no_of_points = x.size()[2] 
      x = self.pointnet(x)
      x=  F.relu(self.bn1(self.conv1(x)))
      x=  F.relu(self.bn2(self.conv2(x)))
      x=  F.relu(self.bn3(self.conv3(x)))
      x= self.conv4(x)
      x = x.transpose(2,1).contiguous()
      x = F.log_softmax(x.view(-1,self.k), dim=-1)
      x = x.view(batchsize, no_of_points, self.k)
      return x