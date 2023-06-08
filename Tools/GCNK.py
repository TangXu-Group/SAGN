from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import math
class Graph2dConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        block_num,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode='zeros',
        adj_mask = None
    ):
        super(Graph2dConvolution, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups,
                                bias,
                                padding_mode)
        
        self.in_features = in_channels
        self.out_features = out_channels
        
        self.W = Parameter(torch.randn(in_channels, in_channels))

        self.reset_parameters()
        self.block_num = block_num
        self.adj_mask = adj_mask
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, input, index):
        index = nn.UpsamplingNearest2d(size = (input.shape[2],input.shape[3]))(index.float()).long()
        
        batch_size = input.shape[0]
        channels = input.shape[1]

        # get one-hot label
        index_ex = torch.zeros(batch_size,self.block_num,input.shape[2],input.shape[3]).cuda()
        index_ex = index_ex.scatter_(1, index, 1)
        block_value_sum = torch.sum(index_ex,dim = (2,3))

        # computing the regional mean of input
        input_ = input.repeat(self.block_num,1,1,1,1).permute(1,0,2,3,4)
        index_ex = index_ex.unsqueeze(2)
        input_means = torch.sum(index_ex * input_,dim = (3,4))/(block_value_sum+(block_value_sum==0).float()).unsqueeze(2) #* mask.unsqueeze(2)

        # computing the adjance metrix
        input_means_ = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
        input_means_ = (input_means_ - input_means.unsqueeze(1)).permute(0, 2, 1, 3)
        M = (self.W).mm(self.W.T)
        adj = input_means_.reshape(batch_size, -1, channels).matmul(M)
        adj = torch.sum(adj * input_means_.reshape(batch_size, -1, channels),dim=2).view(batch_size, self.block_num,self.block_num)
        adj = torch.exp(-1 * adj)#+ torch.eye(self.block_num).repeat(batch_size, 1, 1).cuda()
        if self.adj_mask is not None:
            adj = adj * self.adj_mask

        # generating the adj_mean
        adj_means = input_means.repeat(self.block_num,1,1,1).permute(1,0,2,3) * adj.unsqueeze(3)
        adj_means = (1-torch.eye(self.block_num).reshape(1,self.block_num,self.block_num,1).cuda()) * adj_means
        adj_means = torch.sum(adj_means, dim=2) # batch_size，self.block_num, channel_num

        #obtaining the graph update features
        features = torch.sum(index_ex * (input_ + adj_means.unsqueeze(3).unsqueeze(4)),dim=1)
        
        features = self.Conv2d(features)
        return features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'