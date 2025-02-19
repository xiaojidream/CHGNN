import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init
from dgl.nn.pytorch.glob import GlobalAttentionPooling

class GCN(nn.Module):
    def __init__(self, nfeat=32, nhid1=16, nhid12=8, activation='LeakyRelu', mask_learning=True, nclass=2):
        super(GCN, self).__init__()
        if mask_learning:
            inter_channels = 8
            self.conv_a = nn.Conv1d(nfeat, inter_channels, 1)
            self.conv_b = nn.Conv1d(nfeat, inter_channels, 1)
            self.inter_channels = inter_channels
            self.soft = nn.Softmax(dim=1)

        self.mask_learning = mask_learning
        self.gc1 = GraphConvolution_cat(nfeat, nhid1)
        self.gc2 = GraphConvolution_cat(nhid1, nhid12)
        self.pool = GlobalAttentionPooling(torch.nn.Linear(nhid12, nhid12))


        if activation =='Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()
        elif activation == 'LeakyRelu':
            self.activation = nn.LeakyReLU()

        self.fc = nn.Sequential(
            nn.Linear(nhid12, nclass))
            # nn.Sigmoid())

    def forward(self, x, adj, edge_attr):  # x:[20,50,33]  adj:[20,50,50]
        # 第一个元素代表节点类型
        node_types = x[..., 0]
        x = x[..., 1:]

        Fq = self.conv_a(x.transpose(1, 2)).transpose(1, 2)
        Fk = self.conv_b(x.transpose(1, 2))
        # 特征相似度
        S = self.soft(torch.matmul(Fk, Fk))
        C = S + edge_attr
        # 边与系数矩阵逐元素相乘
        W = torch.mul(adj, C)
        x = self.activation(self.gc1(x, W))
        x = self.activation(self.gc2(x, W))

        unique_types = torch.unique(node_types)
        type_features = []
        for t in unique_types:
            mask = (node_types == t).unsqueeze(-1)
            type_x = x * mask
            type_features.append(self.pool(type_x))

        x_pool = torch.mean(torch.stack(type_features, dim=1), dim=1).values # x:[20,32]




        output = self.fc(x_pool)  # x:[20,1]
        return output


class GraphConvolution_cat(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_cat, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features)) # in_feature: C
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        N, V, C = x.size()
        support = torch.matmul(adj, x)
        support = support.view(N, V, C)
        output = torch.einsum('nik, kj -> nij', support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

