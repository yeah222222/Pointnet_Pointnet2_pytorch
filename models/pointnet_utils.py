import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x): #x.size()=B,D,N
        batchsize = x.size()[0] #B
        x = F.relu(self.bn1(self.conv1(x))) #D*N->64*N
        x = F.relu(self.bn2(self.conv2(x)))  #64*N->128*N
        x = F.relu(self.bn3(self.conv3(x)))#128*N->1024*N
        x = torch.max(x, 2, keepdim=True)[0] #取每一列最大的那个，因为相当于是升维之后在1024维中取出来最大的那个，说明在这个特征中起到最大作用的点的对应的特征值
        x = x.view(-1, 1024)#展平操作，1024列，应该是原来是三维的变成二维了【【1024个数】】

        x = F.relu(self.bn4(self.fc1(x)))#全连接层 1024个特征变为512个
        x = F.relu(self.bn5(self.fc2(x)))#全连接层 512个特征变为256个
        x = self.fc3(x)#最后把256个特征值映射到样本标记空间？全连接层作用是啥？不理解啊

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1) 
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden #为了把每一个x加上[1, 0, 0, 0, 1, 0, 0, 0, 1]
        x = x.view(-1, 3, 3) #展平为一个3*3的什么呢?这有什么用啊？
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k) #balabala*k*k的，这有什么用啊？
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1) #B,N,D
        if D > 3:
            feature = x[:, :, 3:] #xyz法向量
            x = x[:, :, :3] #xyz前三个值
        x = torch.bmm(x, trans) #x是N*3，trans是上面那个3*3的STK3d，这里就是一个B*N*3,每一行变成了每个点的特征了
        if D > 3:
            x = torch.cat([x, feature], dim=2)#以列的形式拼接。将B*N*3作为前面的，然后后面是feature (N*3)
        x = x.transpose(2, 1)#B*3*N
        x = F.relu(self.bn1(self.conv1(x)))#B*64*N

        if self.feature_transform:
            trans_feat = self.fstn(x)#B*64*64
            x = x.transpose(2, 1)#B*N*64
            x = torch.bmm(x, trans_feat)#B*N*64
            x = x.transpose(2, 1)#B*64*N
        else:
            trans_feat = None

        pointfeat = x #B*64*N
        x = F.relu(self.bn2(self.conv2(x)))#B*128*N
        x = self.bn3(self.conv3(x))#B*1024*N
        x = torch.max(x, 2, keepdim=True)[0]#B*1024*1
        x = x.view(-1, 1024)#B*1*1024
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)#B*1024*N
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans): #带有正则化的loss函数
    
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
